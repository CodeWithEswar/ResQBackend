"""Train real YOLO weights, compare held-out metrics, and optionally promote atomically."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ml.artifacts import digest, read_bundle, write_bundle

ROOT = Path(__file__).resolve().parents[1]


def measurements(results):
    return {"precision": float(results.box.mp), "recall": float(results.box.mr), "map50": float(results.box.map50), "map50_95": float(results.box.map)}


def detection_checkpoint(path):
    source = YOLO(str(path))
    if source.task == 'detect':
        return path
    if source.task != 'pose' or source.model.yaml.get('kpt_shape') != [5, 3]:
        raise ValueError('Expected a face detection or five-landmark face checkpoint')
    # The published face model includes an extra landmark branch. Official WIDER
    # labels contain boxes only. Copy all shared detection weights into a Detect
    # head and omit that branch rather than inventing landmark training labels.
    cfg = deepcopy(source.model.yaml)
    cfg.pop('kpt_shape', None)
    cfg['yaml_file']='yolov8n-face-detect.yaml'
    cfg['head'][-1][2:] = ['Detect', ['nc']]
    converted = DetectionModel(cfg, nc=1, verbose=False)
    converted.load(source.model)
    converted.names = {0: 'face'}
    converted.args = {**source.model.args, 'task': 'detect'}
    target = ROOT/'work'/'yolov8n-face-detection-base.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    source.model=converted
    source.task='detect'
    source.ckpt={'train_args': converted.args}
    source.save(str(target))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/wider/yolo/data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/yolov8n-face.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", default="wider-face")
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--freeze",type=int,default=0)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("At least one epoch is required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": args.device}), flush=True)
    manifest = json.loads(args.data.with_name("manifest.json").read_text())
    if min(manifest["splits"][s]["images"] for s in ("train", "val")) < 20:
        raise ValueError("At least 20 genuine images per split are required")
    common = dict(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device, workers=0, plots=False, verbose=False)
    base_weights = detection_checkpoint(args.weights)
    baseline = measurements(YOLO(str(base_weights)).val(**common, project=str(ROOT / "runs"), name=args.name + "-baseline"))
    model = YOLO(str(base_weights))
    model.train(**common, epochs=args.epochs, seed=42, deterministic=True, project=str(ROOT / "runs"), name=args.name,
                optimizer="AdamW", lr0=args.lr, freeze=args.freeze, warmup_epochs=0, warmup_bias_lr=args.lr,
                patience=15, close_mosaic=0, degrees=10, translate=0.08,
                scale=0.3, hsv_v=0.3, fliplr=0.5, flipud=0, amp=False, exist_ok=False)
    best = Path(model.trainer.best)
    candidate = measurements(YOLO(str(best)).val(**common, project=str(ROOT / "runs"), name=args.name + "-evaluation"))
    passed = (candidate["map50"] >= max(0.50, baseline["map50"] - 0.01)
              and candidate["map50_95"] >= baseline["map50_95"]-0.01
              and candidate["precision"] >= baseline["precision"]-0.03
              and candidate["recall"] >= max(0.50, baseline["recall"] - 0.03))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "epochs_requested": args.epochs,
              "weights": best.relative_to(ROOT).as_posix(), "weights_sha256": digest(best), "initial_weights_sha256": digest(args.weights),
              "training_parameters":{"lr":args.lr,"freeze":args.freeze,"batch":args.batch,"imgsz":args.imgsz,"seed":42,"warmup_epochs":0},
              "dataset_manifest_sha256": digest(args.data.with_name("manifest.json")),
              "train_images": manifest["splits"]["train"]["images"], "validation_images": manifest["splits"]["val"]["images"],
              "baseline": baseline, "candidate": candidate, "promotion_gate_passed": passed,
              "disaster_validated": False, "identity_accuracy_evaluated": False,
              "scope": "WIDER FACE detection validation; no claim of disaster identification accuracy"}
    report_path = Path(model.trainer.save_dir) / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.promote:
        if not passed:
            raise RuntimeError("Candidate failed detection gates; deployed weights were preserved")
        bundle = read_bundle(ROOT / "models/pipeline.pkl")
        filename = "resq-face-" + digest(best)[:12] + ".pt"
        shutil.copy2(best, ROOT / "models" / filename)
        bundle["detector"].update(file=filename, sha256=digest(best), local_training=True, training=report)
        # A different detector changes face crops and coverage. Its predecessor's
        # recognition evaluation must not be advertised as current calibration.
        bundle['calibration'] = {'status': 'requires_recalibration', 'ear_recognition_enabled': False,
                                 'previous_evaluation': bundle['calibration'].get('evaluation')}
        write_bundle(ROOT / "models/pipeline.pkl", bundle)
        print("Promoted validated detector. The API activates the verified manifest on the next model request.", flush=True)


if __name__ == "__main__":
    main()
