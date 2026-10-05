"""Fetch the LFW verification benchmark using scikit-learn's published checksums."""
import json
from pathlib import Path
import tarfile
from ml.fetch_assets import download
from ml.artifacts import digest

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'lfw.tgz': ('5976018', '055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0'),
    'pairsDevTrain.txt': ('5976012', '1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa'),
    'pairsDevTest.txt': ('5976009', '7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c'),
}


def main():
    raw = ROOT / 'data/lfw'
    for name, (file_id, expected) in FILES.items():
        target = raw / name
        download('https://ndownloader.figshare.com/files/' + file_id, target)
        if digest(target) != expected:
            raise ValueError('LFW checksum mismatch: ' + name)
    if not (raw / 'lfw').exists():
        with tarfile.open(raw / 'lfw.tgz', 'r:gz') as archive:
            archive.extractall(raw, filter='data')
    (raw / 'sources.json').write_text(json.dumps({
        'dataset': 'Labeled Faces in the Wild', 'purpose': 'research verification benchmark; not a disaster dataset',
        'source': 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/_lfw.py',
        'files': {name: {'sha256': expected, 'url': 'https://ndownloader.figshare.com/files/' + file_id} for name, (file_id, expected) in FILES.items()},
        'usage': 'LFW benchmark terms; no claim of commercial rights to source photographs'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
