"""YOLO face detection, landmark alignment and SFace recognition."""
from pathlib import Path
import os
import tempfile
import threading
import cv2
import numpy as np
import torch

# In serverless environments (Vercel/Lambda), ensure Ultralytics writes to /tmp
yolo_dir = Path(tempfile.gettempdir()) / "Ultralytics"
yolo_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))

from ml.artifacts import read_bundle
from ultralytics import YOLO, settings as yolo_settings

ROOT = Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)
torch.set_num_threads(4)
yolo_settings.update({'sync':False,'clearml':False,'comet':False,'dvc':False,'mlflow':False,'raytune':False,'wandb':False})

def unit(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(x)
    if not np.isfinite(x).all() or norm < 1e-8:
        raise ValueError('No usable visual signal')
    return x / norm

def gray(image, size=(64, 128)):
    if image is None or image.size == 0:
        raise ValueError('Invalid image')
    value = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(value, size, interpolation=cv2.INTER_AREA)

_hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
def hog(image):
    return unit(_hog.compute(gray(image)))

def quality(image):
    value = gray(image)
    return {'brightness': float(value.mean()/255), 'contrast': float(value.std()/255),
            'laplacian_variance': float(cv2.Laplacian(value, cv2.CV_64F).var())}

class Engine:
    def __init__(self, models=None):
        models = Path(models or ROOT / 'models')
        self.bundle = read_bundle(models / 'pipeline.pkl')
        self.recognizer = cv2.FaceRecognizerSF.create(str(models/self.bundle['recognizer']['file']), '')
        self.landmarks = cv2.FaceDetectorYN.create(str(models/self.bundle['landmarks']['file']), '', (320,320), 0.65, 0.3, 5000)
        self.detector = YOLO(str(models/self.bundle['detector']['file']), task='detect')
        if self.detector.names != {0: 'face'}:
            raise ValueError('A face-trained YOLO checkpoint is required, not a general person detector')
        self.lock = threading.RLock()
        # Warm both model paths at startup, rather than delaying the first user request.
        with self.lock:
            self.detector.predict(np.zeros((640,640,3), dtype=np.uint8), device='cpu', imgsz=640, verbose=False)

    def face_crop(self, image):
        # The legacy cropped flag cannot bypass face detection or alignment.
        return self.face(image)

    def detect(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        if width < 20 or height < 20:
            raise ValueError('Image too small')
        scale = min(1, 1600 / max(width, height))
        sample = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale)
        with self.lock:
            boxes = self.detector.predict(sample, device='cpu', imgsz=640, conf=0.4, iou=0.45, max_det=100, verbose=False)[0].boxes
            raw_boxes = boxes.xyxy.cpu().numpy() / scale
            confidence = boxes.conf.cpu().numpy()
        found = []
        for box, score in zip(raw_boxes, confidence):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(width,x2), min(height,y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            q = quality(crop)
            q['face_width'] = int(x2-x1)
            q['face_height'] = int(y2-y1)
            reasons = self.quality_reasons(q)
            found.append({'box': [int(x1),int(y1),int(x2),int(y2)], 'confidence': float(score),
                          'quality': q, 'usable': not reasons, 'issues': reasons})
        found.sort(key=lambda f: (f['box'][0],f['box'][1],f['box'][2],f['box'][3]))
        return [{**face, 'index': index} for index, face in enumerate(found)]

    def quality_reasons(self, value):
        limits = self.bundle['quality']
        reasons = []
        if min(value['face_width'],value['face_height']) < limits['min_face_pixels']:
            reasons.append('Face is too small; move closer or use a higher-resolution photo.')
        if value['contrast'] < limits['min_contrast']:
            reasons.append('Face has insufficient contrast.')
        if not limits['min_brightness'] <= value['brightness'] <= limits['max_brightness']:
            reasons.append('Face lighting is too dark or overexposed.')
        if value['laplacian_variance'] < limits['min_laplacian_variance']:
            reasons.append('Face is blurred; use a sharper photo.')
        return reasons

    def extract_face(self, image, face_index=None):
        found = self.detect(image)
        if not found:
            raise ValueError('No face detected. Use a clearer face photograph.')
        if face_index is None:
            if len(found) != 1:
                raise ValueError('Multiple faces detected. Choose a face before searching or enrolling.')
            face_index = 0
        if face_index < 0 or face_index >= len(found):
            raise ValueError('Selected face is not present in this photo. Detect the faces again.')
        chosen = found[face_index]
        if not chosen['usable']:
            raise ValueError(' '.join(chosen['issues']))
        x1,y1,x2,y2 = chosen['box']
        margin = int(max(x2-x1,y2-y1)*0.25)
        roi = image[max(0,y1-margin):min(image.shape[0],y2+margin),max(0,x1-margin):min(image.shape[1],x2+margin)]
        origin_x,origin_y = max(0,x1-margin),max(0,y1-margin)
        roi_scale = min(1,1024/max(roi.shape[:2]))
        if roi_scale < 1:
            roi=cv2.resize(roi,None,fx=roi_scale,fy=roi_scale)
        with self.lock:
            self.landmarks.setInputSize((roi.shape[1],roi.shape[0]))
            _, candidates = self.landmarks.detect(roi)
            if candidates is None:
                raise ValueError('Facial landmarks are obscured. Use a clearer, front-facing photograph.')
            target = np.array([x1-origin_x,y1-origin_y,x2-origin_x,y2-origin_y])*roi_scale
            def overlap(row):
                box = np.array([row[0],row[1],row[0]+row[2],row[1]+row[3]])
                intersection = np.maximum(0,np.minimum(target[2:],box[2:])-np.maximum(target[:2],box[:2])).prod()
                union = np.prod(target[2:]-target[:2]) + row[2]*row[3] - intersection
                return float(intersection / max(union,1))
            row = max(candidates, key=overlap)
            if overlap(row) < 0.3:
                raise ValueError('Could not align the selected face reliably. Use another photograph.')
            aligned = self.recognizer.alignCrop(roi,row)
            vector = unit(self.recognizer.feature(aligned))
        return vector, {**chosen['quality'], 'detector_confidence': chosen['confidence'], 'box': chosen['box'],
                        'face_index': face_index, 'aligned': True}

    def face(self, image):
        return self.extract_face(image)[0]

    def describe(self, image, modality, cropped=False):
        if modality == 'face':
            return self.face_crop(image) if cropped else self.face(image)
        if modality == 'ear':
            return hog(image)
        raise ValueError('Unsupported modality')

MODEL_IDS = {'face': 'opencv-sface-2021dec-128', 'ear': 'opencv-hog-64x128-3780-v1'}

def rank_candidates(query, references, thresholds, top_k=5):
    """Exact cosine search; maximum across a person's references per modality."""
    if 'face' not in query or not references:
        return []
    scores = {}
    for ref in references:
        modality = ref['modality']
        # HOG ear similarity has no validated identity model and cannot create a lead.
        if modality != 'face':
            continue
        q, r = unit(query[modality]), np.asarray(ref['embedding'], dtype=np.float32)
        if q.shape != r.shape or ref['model'] != MODEL_IDS[modality]:
            raise ValueError('Incompatible model or descriptor dimension')
        r = unit(r)
        similarity = float(np.clip(np.dot(q, r), -1, 1))
        parts = scores.setdefault(ref['person_id'], {})
        parts[modality] = max(similarity, parts.get(modality, -1))
    candidates = []
    for person_id, parts in scores.items():
        score = parts['face']
        if score < thresholds['face']:
            continue
        candidates.append({'person_id':person_id, 'score':score, 'components':parts,
                           'status':'pending', 'score_is_probability':False})
    return sorted(candidates, key=lambda c:(-c['score'], c['person_id']))[:top_k]
