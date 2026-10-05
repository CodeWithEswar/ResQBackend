"""Create a deployable model archive without including case photos or training data."""
import argparse
from pathlib import Path
import zipfile
from ml.artifacts import digest, read_bundle

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'work/resq-model.zip')
    args=parser.parse_args()
    models=ROOT/'models'
    bundle=read_bundle(models/'pipeline.pkl')
    files=[models/'pipeline.pkl',models/'pipeline.json']
    files.extend(models/bundle[k]['file'] for k in ('detector','recognizer','landmarks'))
    files.extend((models/'licenses').glob('*'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(models))
    print(f'Model archive: {args.output}\nSHA256: {digest(args.output)}')


if __name__=='__main__':
    main()
