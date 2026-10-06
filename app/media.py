"""Bounded video decoding and displayable, numbered face-detection frames."""
import base64
import math
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel


class FaceQuality(BaseModel):
    brightness: float
    contrast: float
    laplacian_variance: float
    face_width: int
    face_height: int


class DetectedFace(BaseModel):
    index: int
    box: tuple[int, int, int, int]
    confidence: float
    quality: FaceQuality
    usable: bool
    issues: list[str]


class FramePreview(BaseModel):
    image: str
    annotated_image: str
    mime_type: str = 'image/jpeg'
    width: int
    height: int


class DetectionResult(BaseModel):
    width: int
    height: int
    faces: list[DetectedFace]
    selection_required: bool
    detector: str
    detector_version: str
    processing_ms: int
    frame: FramePreview | None = None


class LiveDetectionResult(DetectionResult):
    sequence: int
    captured_at_ms: int | None = None


class VideoFrame(DetectionResult):
    frame_index: int
    timestamp_seconds: float


class VideoDetectionResult(BaseModel):
    duration_seconds: float
    fps: float
    total_frames: int
    sampled_frames: int
    frames_with_faces: int
    sampling_interval_seconds: float
    frames: list[VideoFrame]
    detector: str
    detector_version: str
    processing_ms: int
    message: str
    warnings: list[str]


def jpeg_uri(image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(422, 'The detected frame could not be rendered.')
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')


def resize_frame(image, max_edge=1600):
    height, width = image.shape[:2]
    scale = min(1, max_edge / max(height, width))
    return image if scale == 1 else cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def preview_frame(image, faces):
    sample = resize_frame(image)
    height, width = sample.shape[:2]
    scale_x, scale_y = width / image.shape[1], height / image.shape[0]
    annotated = sample.copy()
    stroke = max(2, round(max(width, height) / 450))
    font_scale = max(0.5, min(1.1, max(width, height) / 1100))
    for face in faces:
        x1, y1, x2, y2 = face['box']
        x1, x2 = round(x1 * scale_x), round(x2 * scale_x)
        y1, y2 = round(y1 * scale_y), round(y2 * scale_y)
        color = (155, 224, 94) if face['usable'] else (149, 164, 253)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, stroke)
        label = f"Face {face['index'] + 1}  {face['confidence']:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        left = min(x1, max(0, width - label_w - 8))
        top = max(0, y1 - label_h - baseline - 8)
        cv2.rectangle(annotated, (left, top), (min(width - 1, left + label_w + 8), top + label_h + baseline + 8), color, -1)
        cv2.putText(annotated, label, (left + 4, top + label_h + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
    return {'image': jpeg_uri(sample), 'annotated_image': jpeg_uri(annotated),
            'mime_type': 'image/jpeg', 'width': width, 'height': height}


def detect_frame(image, engine, include_frame=True):
    started = time.monotonic()
    try:
        faces = engine.detect(image)
        preview = preview_frame(image, faces) if include_frame else None
    except (ValueError, cv2.error):
        raise HTTPException(422, 'Faces could not be detected in this frame. Choose a clear, correctly encoded image.') from None
    return {'width': image.shape[1], 'height': image.shape[0], 'faces': faces,
            'selection_required': len(faces) > 1, 'detector': engine.bundle['detector']['architecture'],
            'detector_version': engine.bundle['detector']['sha256'], 'frame': preview,
            'processing_ms': round((time.monotonic() - started) * 1000)}


def video_suffix(header):
    # Do not accept URLs, HLS playlists or arbitrary files as video sources.
    if header[4:8] == b'ftyp':
        return '.mp4'
    if header[:4] == b'RIFF' and header[8:12] == b'AVI ':
        return '.avi'
    if header[:4] == b'\x1a\x45\xdf\xa3':
        return '.webm'
    raise HTTPException(415, 'Choose an MP4, MOV, AVI or WebM video with a supported video codec.')


def detect_video(upload, engine, limits, interval_seconds, max_frames, result_limit):
    started = time.monotonic()
    upload.seek(0)
    header = upload.read(256)
    if not header:
        raise HTTPException(422, 'Choose a video first.')
    suffix = video_suffix(header)
    if len(header) > limits.max_video_bytes:
        raise HTTPException(413, f'Choose a video smaller than {limits.max_video_bytes // (1024 * 1024)} MB.')
    cap = None
    # A unique private temporary file is removed on success, errors and timeout.
    with tempfile.TemporaryDirectory(prefix='resq-video-') as temporary:
        path = Path(temporary) / ('upload' + suffix)
        total_bytes = len(header)
        with path.open('wb') as output:
            output.write(header)
            while chunk := upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > limits.max_video_bytes:
                    raise HTTPException(413, f'Choose a video smaller than {limits.max_video_bytes // (1024 * 1024)} MB.')
                output.write(chunk)
        try:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            if not cap.isOpened():
                raise HTTPException(415, 'This video cannot be decoded. Export it as an MP4 with H.264 video and try again.')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
                raise HTTPException(422, 'The video has invalid timing or dimensions. Export a fresh MP4 and try again.')
            if width * height > limits.max_pixels:
                raise HTTPException(413, 'Video frame dimensions exceed the allowed image limit.')
            total_frames = round(count)
            duration = total_frames / fps
            if duration > limits.max_video_seconds:
                raise HTTPException(413, f'Choose a clip no longer than {limits.max_video_seconds:g} seconds.')
            # Spread a bounded set of samples across the whole clip. End-frame
            # detections remain eligible rather than silently truncating a video.
            requested = max(1, math.ceil(duration / interval_seconds))
            sample_count = min(requested, max_frames, total_frames)
            indices = np.unique(np.rint(np.linspace(0, total_frames - 1, sample_count)).astype(int))
            frames = []
            with_faces = 0
            warnings = []
            if requested > sample_count:
                warnings.append('Sampling was spread across the clip to stay within the frame limit. Brief appearances between samples can be missed.')
            sampled_indices = []
            for index in indices:
                if time.monotonic() - started > limits.video_processing_seconds:
                    if frames:
                        warnings.append('Video processing reached the time limit. Returning the clearest face frames analyzed so far.')
                        break
                    raise HTTPException(503, 'Video analysis exceeded the processing limit. Trim the clip or analyze fewer frames.', headers={'Retry-After': '3'})
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                    raise HTTPException(422, 'This video does not support reliable frame seeking. Export a fresh MP4 and try again.')
                ok, image = cap.read()
                if not ok or image is None:
                    raise HTTPException(422, 'A video frame could not be decoded. Export a fresh MP4 and try again.')
                if image.shape[0] * image.shape[1] > limits.max_pixels:
                    raise HTTPException(413, 'Video frame dimensions exceed the allowed image limit.')
                sampled_indices.append(int(index))
                image = resize_frame(image)
                detected = detect_frame(image, engine, include_frame=False)
                if detected['faces']:
                    with_faces += 1
                    usable = [face for face in detected['faces'] if face['usable']]
                    score = (bool(usable), max(face['quality']['laplacian_variance'] for face in usable or detected['faces']))
                    if len(frames) < result_limit or score > min(frame['_score'] for frame in frames):
                        encoded = jpeg_uri(image)
                        detected['frame'] = preview_frame(image, detected['faces'])
                        detected['frame']['image'] = encoded
                        detected.update(frame_index=int(index), timestamp_seconds=round(int(index) / fps, 3), _score=score)
                        frames.append(detected)
                        frames.sort(key=lambda frame: frame['_score'], reverse=True)
                        frames = frames[:result_limit]
            frames.sort(key=lambda frame: frame['frame_index'])
            for frame in frames:
                frame.pop('_score')
            final_indices = sampled_indices if sampled_indices else indices
            effective_interval = (int(final_indices[-1]) - int(final_indices[0])) / fps / (len(final_indices) - 1) if len(final_indices) > 1 else 0
            return {'duration_seconds': round(duration, 3), 'fps': fps, 'total_frames': total_frames,
                    'sampled_frames': len(final_indices), 'frames_with_faces': with_faces, 'frames': frames,
                    'sampling_interval_seconds': round(effective_interval, 3),
                    'detector': engine.bundle['detector']['architecture'], 'detector_version': engine.bundle['detector']['sha256'],
                    'processing_ms': round((time.monotonic() - started) * 1000),
                    'message': 'Select a detected frame, then choose a face to compare.' if frames else 'No faces were detected in the sampled frames. Try a closer, clearer clip.',
                    'warnings': warnings}
        except cv2.error:
            raise HTTPException(415, 'The video could not be decoded. Export an MP4 with H.264 video and try again.') from None
        finally:
            if cap is not None:
                cap.release()
