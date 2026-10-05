"""Download published checkpoints and official WIDER FACE archives, recording hashes."""
import argparse
import json
import os
from pathlib import Path
import httpx
from ml.artifacts import digest, write_bundle

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
WEIGHTS_SHA256 = "d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca"
DATA_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing {destination.name}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90, connect=30)) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            count = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
                    count += len(chunk)
                    if count % (100 * 1024 * 1024) < len(chunk):
                        print(f"{destination.name}: {count // (1024 * 1024)} MB", flush=True)
    os.replace(temporary, destination)
    print(f"Downloaded {destination.name} ({count:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true")
    args = parser.parse_args()
    models = ROOT / "models"
    download(WEIGHTS_URL, models / "yolov8n-face.pt")
    if digest(models / "yolov8n-face.pt") != WEIGHTS_SHA256:
        raise ValueError('Published initialization checksum changed; review the new release before using it')
    bundle = {
        "detector": {"file": "yolov8n-face.pt", "sha256": digest(models / "yolov8n-face.pt"),
                     "architecture": "YOLOv8n-face", "source": WEIGHTS_URL, "dataset": "WIDER FACE", "local_training": False},
        "recognizer": {"file": "sface.onnx", "sha256": digest(models / "sface.onnx"), "model_id": "opencv-sface-2021dec-128", "dimensions": 128},
        "landmarks": {"file": "yunet.onnx", "sha256": digest(models / "yunet.onnx")},
        "thresholds": {"face": 0.363},
        "calibration": {"status": "unvalidated_for_disaster_use", "source": "Published SFace cosine reference: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html", "ear_recognition_enabled": False},
        "quality": {"min_face_pixels": 64, "min_contrast": 0.04, "min_brightness": 0.06, "max_brightness": 0.96, "min_laplacian_variance": 15},
        "deployment": {"identity_confirmation": "independent_human_review", "disaster_validated": False},
    }
    if not (models / "pipeline.pkl").exists():
        write_bundle(models / "pipeline.pkl", bundle)
    if args.dataset:
        raw = ROOT / "data" / "wider" / "raw"
        for name in ("wider_face_split.zip", "WIDER_train.zip", "WIDER_val.zip"):
            download(DATA_URL + name, raw / name)
        manifest = {"source": "https://huggingface.co/datasets/CUHK-CSE/wider_face", "license": "CC-BY-NC-ND-4.0", "purpose": "face detection research; no identity labels", "archives": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in raw.glob("*.zip")}}
        (raw / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
