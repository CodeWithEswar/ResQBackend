"""Calibrate on LFW development pairs; evaluate on disjoint development-test subjects."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import cv2
import numpy as np
from app.engine import Engine
from ml.artifacts import digest, read_bundle, write_bundle

ROOT = Path(__file__).resolve().parents[1]


def pairs(path):
    result = []
    for line in Path(path).read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) == 3:
            name, first, second = fields
            result.append((name, int(first), name, int(second), True))
        elif len(fields) == 4:
            first_name, first, second_name, second = fields
            result.append((first_name, int(first), second_name, int(second), False))
        else:
            raise ValueError('Malformed LFW verification pair')
    return result


def balanced(values, limit):
    rng = random.Random(42)
    positives = [p for p in values if p[4]]
    negatives = [p for p in values if not p[4]]
    rng.shuffle(positives); rng.shuffle(negatives)
    return positives[:limit] + negatives[:limit] if limit else positives + negatives


def metrics(scores, threshold):
    positives = np.array([s['score'] for s in scores if s['same']])
    negatives = np.array([s['score'] for s in scores if not s['same']])
    if not len(positives) or not len(negatives):
        raise ValueError('Both genuine and impostor pairs are required')
    false_accepts = int(np.count_nonzero(negatives >= threshold))
    n = len(negatives); p = false_accepts/n; z=1.96
    upper = (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    return {'genuine_pairs': len(positives), 'impostor_pairs': n, 'true_accept_rate': float(np.mean(positives >= threshold)),
            'false_accept_rate': p, 'false_accepts': false_accepts, 'far_wilson_95_upper': upper}


def calibrate(scores, target_far):
    negatives = np.array([s['score'] for s in scores if not s['same']])
    if len(negatives) < 100:
        raise ValueError('At least 100 usable impostor calibration pairs are required')
    possibilities = sorted({-1., 1., *(s['score'] for s in scores), *(float(np.nextafter(s, np.inf)) for s in negatives)})
    for threshold in possibilities:
        if metrics(scores, threshold)['false_accept_rate'] <= target_far:
            return float(threshold)
    raise ValueError('No threshold meets the requested false-accept rate')


def score_pairs(values, engine, data, cache):
    results = []; rejected = []
    for index, (first_name, first, second_name, second, same) in enumerate(values):
        paths = [data/'lfw'/name/f'{name}_{number:04d}.jpg' for name, number in ((first_name,first),(second_name,second))]
        vectors = []
        for path in paths:
            if path not in cache:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('Missing LFW image: ' + str(path))
                try:
                    # LFW labels the central portrait; background faces can remain
                    # in the original image. Use the published LFW evaluation crop
                    # as explicit selection metadata, as a user selects a face in
                    # the app. Do not guess an identity for every face in a scene.
                    faces=engine.detect(image)
                    height,width=image.shape[:2]
                    target=np.array([78/250*width,70/250*height,172/250*width,195/250*height])
                    def overlap(face):
                        box=np.array(face['box'])
                        intersection=np.maximum(0,np.minimum(box[2:],target[2:])-np.maximum(box[:2],target[:2])).prod()
                        return float(intersection/max(1,np.prod(box[2:]-box[:2])+np.prod(target[2:]-target[:2])-intersection))
                    if not faces:
                        raise ValueError('No labeled central face detected')
                    selected=max(faces,key=overlap)
                    if overlap(selected)<0.3:
                        raise ValueError('The labeled central face could not be selected reliably')
                    cache[path] = engine.extract_face(image,selected['index'])[0]
                except ValueError as error:cache[path] = str(error)
            vectors.append(cache[path])
        if any(isinstance(v,str) for v in vectors):
            rejected.append({'pair': index, 'same': same, 'reasons': [v for v in vectors if isinstance(v,str)]})
        else:
            results.append({'pair': index, 'same': same, 'score': float(np.clip(np.dot(*vectors),-1,1))})
        if (index+1) % 100 == 0:
            print(f'Processed {index+1}/{len(values)} pairs; usable={len(results)}', flush=True)
    return results, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/lfw')
    parser.add_argument('--train-per-class', type=int, default=250)
    parser.add_argument('--test-per-class', type=int, default=150)
    parser.add_argument('--target-far', type=float, default=0.01)
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    if not 0 < args.target_far < 1:
        parser.error('Target false-accept rate must be between zero and one')
    train = balanced(pairs(args.data/'pairsDevTrain.txt'), args.train_per_class)
    subjects = {name for p in train for name in (p[0],p[2])}
    test = balanced([p for p in pairs(args.data/'pairsDevTest.txt') if p[0] not in subjects and p[2] not in subjects], args.test_per_class)
    if not test:
        raise ValueError('No subject-disjoint test pairs remain')
    engine=Engine(); cache={}
    training, train_rejected = score_pairs(train,engine,args.data,cache)
    testing, test_rejected = score_pairs(test,engine,args.data,cache)
    # Do not lower below the published SFace cosine reference on a small sample.
    # https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
    threshold = max(0.363,calibrate(training,args.target_far))
    report = {'created_at':datetime.now(timezone.utc).isoformat(), 'dataset':'LFW original photographs and official development pairs',
              'detector_sha256':engine.bundle['detector']['sha256'], 'recognizer_sha256':engine.bundle['recognizer']['sha256'],
              'threshold':threshold, 'target_calibration_far':args.target_far, 'calibration':metrics(training,threshold),
              'minimum_sface_reference_threshold':0.363,
              'face_selection':'LFW central portrait, using the scikit-learn evaluation region; equivalent to explicit app face selection',
              'test':metrics(testing,threshold), 'previous_threshold_test':metrics(testing,engine.bundle['thresholds']['face']),
              'subject_disjoint':True, 'train_pairs_attempted':len(train), 'test_pairs_attempted':len(test),
              'train_rejected':len(train_rejected), 'test_rejected':len(test_rejected),
              'pair_files':{name:digest(args.data/name) for name in ('pairsDevTrain.txt','pairsDevTest.txt')},
              'disaster_validated':False, 'limitations':'LFW research sample, not injured, deceased, occluded disaster victims; finite-sample FAR uncertainty. Rejected inputs count against coverage, not recognition denominators.'}
    output=ROOT/'work/lfw-evaluation.json'; output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({**report,'train_scores':training,'test_scores':testing,'rejections':{'train':train_rejected,'test':test_rejected}},indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if args.promote:
        if report['test']['true_accept_rate'] < 0.75 or report['test']['false_accept_rate'] > 0.02 or len(testing) / len(test) < 0.7:
            raise RuntimeError('Recognition gates failed; existing threshold was preserved')
        bundle=read_bundle(ROOT/'models/pipeline.pkl')
        if bundle['detector']['sha256']!=report['detector_sha256']:
            raise RuntimeError('Detector changed during evaluation; re-run calibration with the deployed detector')
        bundle['thresholds']['face']=threshold
        bundle['calibration']={'status':'lfw_research_calibrated','ear_recognition_enabled':False,'evaluation':report}
        write_bundle(ROOT/'models/pipeline.pkl',bundle)
        print('Promoted research-calibrated face threshold; disaster validation remains false.',flush=True)


if __name__ == '__main__':
    main()
