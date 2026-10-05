import os
import sys
import tempfile
from pathlib import Path

# Ensure backend root directory is in sys.path for absolute imports ('app', 'ml')
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Redirect Ultralytics config to writable /tmp directory in serverless environments
yolo_dir = Path(tempfile.gettempdir()) / "Ultralytics"
yolo_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))

from app.main import app
