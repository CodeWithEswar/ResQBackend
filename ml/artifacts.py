"""Versioned, primitive-only pickle manifests; never deserialize uploaded models."""
import hashlib
import io
import json
import os
import pickle
from pathlib import Path

SCHEMA = 1


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Executable pickle objects are forbidden")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent pickle references are forbidden")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_bundle(path, payload):
    # JSON round trip excludes executable Python objects and ndarray pickles.
    payload = json.loads(json.dumps({"schema_version": SCHEMA, **payload}, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=4))
    os.replace(temporary, path)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_bundle(path, verify=True):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 4_000_000:
        raise ValueError("Model manifest is too large")
    try:
        value = PrimitiveUnpickler(io.BytesIO(raw)).load()
        value = json.loads(json.dumps(value, allow_nan=False))
    except (pickle.UnpicklingError,EOFError,TypeError,ValueError) as error:
        raise ValueError('Invalid or executable model manifest') from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported model manifest")
    for key in ("detector", "recognizer", "landmarks"):
        entry = value[key]
        filename = entry["file"]
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("Model files must remain inside the model directory")
        if verify and digest(path.parent / filename) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {key}")
    threshold = value["thresholds"]["face"]
    if not isinstance(threshold, (int, float)) or not -1 <= threshold <= 1:
        raise ValueError("Invalid face threshold")
    return value
