"""Create a user-only WiFi Radar package from a validated factory model.

Run from the repository root:
  python -m athena.package_user --version 1.0.0

The package intentionally excludes factory recordings and training code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "athena"
MODEL_DIR = APP / "models"
MODEL_PATH = MODEL_DIR / "activity_model.pt"
METRICS_PATH = MODEL_DIR / "metrics.json"
REQUIRED_GROUPS = {
    "empty_room": {"empty", "room_empty", "empty_room"},
    "person_static": {"sitting_still", "person_static", "still", "static"},
    "movement": {"movement", "walking", "moving"},
    "fall": {"fall", "falling", "fall_event", "fall_detected", "fall_detection"},
}
RUNTIME_FILES = [
    "athena/__init__.py", "athena/__main__.py", "athena/app.py",
    "athena/data.py", "athena/edition.py", "athena/live.py",
    "athena/model.py", "athena/monitor.py", "athena/proto.py",
    "athena/receiver.py", "athena/static/index.html", "requirements.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model() -> tuple[list[str], dict]:
    if not MODEL_PATH.is_file():
        raise SystemExit("No factory model found. Train a model first: " + str(MODEL_PATH))
    if not METRICS_PATH.is_file():
        raise SystemExit("Test metrics are missing. A model without evaluation metrics cannot be packaged.")
    try:
        import torch
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"Could not load factory model: {exc}") from exc
    labels = [str(x).strip().lower() for x in checkpoint.get("labels", [])]
    missing = [name for name, aliases in REQUIRED_GROUPS.items()
               if not aliases.intersection(labels)]
    if missing:
        raise SystemExit("Model is missing required classes: " + ", ".join(missing))
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read test metrics: {exc}") from exc
    if not isinstance(metrics.get("test"), dict) or not isinstance(metrics["test"].get("per_class"), dict):
        raise SystemExit("Metrics file does not contain per-class test results.")
    # Do not certify model quality automatically: packaging is allowed only after
    # a researcher has reviewed the results and consciously approved release.
    return labels, metrics


def build(version: str, output: Path, researcher_approved: bool) -> Path:
    if not researcher_approved:
        raise SystemExit("Review the independent test results first, then add --researcher-approved.")
    labels, metrics = validate_model()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wifi-radar-user-") as temp:
        stage = Path(temp) / "WiFi-Radar-User"
        for relative in RUNTIME_FILES:
            source = APP / relative
            if not source.is_file():
                raise SystemExit(f"Required runtime file is missing: {source}")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (stage / "models").mkdir(parents=True, exist_ok=True)
        user_model = stage / "models" / "activity_model.pt"
        shutil.copy2(MODEL_PATH, user_model)
        shutil.copy2(METRICS_PATH, stage / "models" / "metrics.json")
        manifest = {
            "product": "WiFi Radar",
            "edition": "user",
            "version": version,
            "model_file": "models/activity_model.pt",
            "model_sha256": sha256(user_model),
            "labels": labels,
            "test_metrics": metrics.get("test"),
            "factory_data_included": False,
            "factory_training_tools_included": False,
            "researcher_approved": True,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (stage / "RUN_USER_EDITION.txt").write_text(
            "WiFi Radar — User Edition\n"
            "Install Python 3.10+ and the packages in requirements.txt.\n"
            "Run: python -m athena --edition user --host 0.0.0.0\n"
            "The local model runs on this processing device. The ESP32-S3 streams CSI over UDP.\n"
            "Room calibration is required before monitoring.\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Package WiFi Radar User Edition")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--output", default=str(ROOT / "dist" / "WiFi-Radar-User.zip"))
    parser.add_argument("--researcher-approved", action="store_true",
                        help="confirm that independent test metrics were reviewed")
    args = parser.parse_args()
    result = build(args.version, Path(args.output), args.researcher_approved)
    print(f"Created user package: {result}")


if __name__ == "__main__":
    main()
