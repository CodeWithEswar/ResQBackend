"""Choose one usable official validation photograph for real inference regression tests."""
import json
from pathlib import Path
import shutil
import cv2
from app.engine import Engine
from ml.artifacts import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/wider/yolo'
    manifest=json.loads((data/'manifest.json').read_text())
    engine=Engine()
    for entry in manifest['splits']['val']['entries']:
        if entry['face_count'] != 1:
            continue
        source=data/'images/val'/(entry['output']+'.jpg')
        image=cv2.imread(str(source))
        try:vector,q=engine.extract_face(image)
        except ValueError:continue
        target=ROOT/'tests/fixtures/face-reference.jpg'
        shutil.copy2(source,target)
        target.with_suffix('.json').write_text(json.dumps({'dataset':'WIDER FACE official validation', 'source':manifest['source'], 'original_file':entry['file'], 'sha256':digest(target), 'license':manifest['license'], 'purpose':'inference regression only; no identity accuracy claim'},indent=2),encoding='utf-8')
        print(json.dumps({'fixture':str(target),'quality':q,'dimensions':len(vector)}))
        return
    raise RuntimeError('No usable single-face fixture in validation selection')


if __name__=='__main__':
    main()
