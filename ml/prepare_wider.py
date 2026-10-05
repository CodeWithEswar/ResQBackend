"""Convert official WIDER splits to YOLO labels without generating pseudo-labels."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def annotations(text):
    lines = iter(text.splitlines())
    for filename in lines:
        if not filename.strip():
            continue
        count = int(next(lines))
        rows = [list(map(float, next(lines).split())) for _ in range(max(count, 1))]
        yield filename, rows if count else []


def select(records, limit, seed):
    # Round robin through shuffled event groups avoids a parade-only prefix sample.
    groups = defaultdict(list)
    for entry in records:
        groups[entry[0].split("/")[0]].append(entry)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    while keys and (not limit or len(result) < limit):
        for key in list(keys):
            result.append(groups[key].pop())
            if not groups[key]:
                keys.remove(key)
            if limit and len(result) == limit:
                break
    return result


def convert(raw, output, train_limit=0, val_limit=0, seed=42):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "WIDER FACE", "source": "https://huggingface.co/datasets/CUHK-CSE/wider_face", "license": "CC-BY-NC-ND-4.0", "seed": seed, "identity_labels": False, "splits": {}}
    hashes = set()
    with zipfile.ZipFile(Path(raw) / "wider_face_split.zip") as labels:
        for split, limit in (("train", train_limit), ("val", val_limit)):
            records = select(list(annotations(labels.read(f"wider_face_split/wider_face_{split}_bbx_gt.txt").decode("utf-8"))), limit, seed)
            images = output / "images" / split
            targets = output / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            targets.mkdir(parents=True, exist_ok=True)
            entries = []
            with zipfile.ZipFile(Path(raw) / f"WIDER_{split}.zip") as archive:
                for filename, rows in records:
                    if ".." in PurePosixPath(filename).parts or PurePosixPath(filename).is_absolute():
                        raise ValueError("Invalid dataset path")
                    data = archive.read(f"WIDER_{split}/images/{filename}")
                    sha = hashlib.sha256(data).hexdigest()
                    if sha in hashes:
                        continue
                    hashes.add(sha)
                    with Image.open(io.BytesIO(data)) as image:
                        width, height = image.size
                    boxes = []
                    for row in rows:
                        if len(row) < 8 or row[7] == 1:
                            continue
                        x, y, w, h = row[:4]
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(width, x + w), min(height, y + h)
                        if x2 - x1 < 4 or y2 - y1 < 4:
                            continue
                        boxes.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                    stem = sha[:24]
                    (images / f"{stem}.jpg").write_bytes(data)
                    (targets / f"{stem}.txt").write_text("\n".join(boxes), encoding="utf-8")
                    entries.append({"file": filename, "output": stem, "sha256": sha, "face_count": len(boxes), "event": filename.split("/")[0]})
            manifest["splits"][split] = {"images": len(entries), "faces": sum(e["face_count"] for e in entries), "entries": entries}
            (output / f"{split}.txt").write_text("\n".join(str(images / (e["output"] + ".jpg")) for e in entries), encoding="utf-8")
    (output / "data.yaml").write_text(f"path: {output.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: face\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in m.items() if k != "entries"} for s, m in manifest["splits"].items()}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/wider/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/wider/yolo")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_limit, args.val_limit) < 0:
        parser.error("Limits cannot be negative")
    convert(args.raw, args.output, args.train_limit, args.val_limit, args.seed)


if __name__ == "__main__":
    main()
