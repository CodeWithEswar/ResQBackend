"""Reproducible face detection training and artifact management."""
import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / 'work'
(_cache / 'ultralytics' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
(_cache / 'matplotlib').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_cache / 'matplotlib'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(_cache / 'ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')
