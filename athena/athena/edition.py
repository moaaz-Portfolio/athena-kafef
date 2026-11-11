"""Edition and local settings helpers for WiFi Radar."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

FACTORY = "factory"
USER = "user"

DEFAULT_NOTIFICATIONS = {
    "fall": True,
    "movement": True,
    "presence": True,
    "empty_room": False,
    "door_presence": True,
    "device_offline": True,
    "csi_quality": False,
}
DEFAULT_OUTPUTS = {"buzzer_enabled": True}


def normalize_edition(value: str | None) -> str:
    edition = (value or FACTORY).strip().lower()
    if edition not in (FACTORY, USER):
        raise ValueError("edition must be 'factory' or 'user'")
    return edition


def load_settings(path: str) -> dict:
    settings = {"notifications": dict(DEFAULT_NOTIFICATIONS),
                "outputs": dict(DEFAULT_OUTPUTS)}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            saved = json.load(stream)
        for section in ("notifications", "outputs"):
            if isinstance(saved.get(section), dict):
                allowed = settings[section]
                for key, value in saved[section].items():
                    if key in allowed and isinstance(value, bool):
                        allowed[key] = value
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        # A damaged settings file must not prevent local monitoring from starting.
        pass
    return settings


def save_settings(path: str, settings: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp",
                                     dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(settings, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def update_preferences(settings: dict, section: str, incoming: dict) -> dict:
    if section not in ("notifications", "outputs") or not isinstance(incoming, dict):
        raise ValueError("invalid settings section")
    allowed = settings[section]
    for key, value in incoming.items():
        if key not in allowed or not isinstance(value, bool):
            raise ValueError(f"invalid setting: {key}")
        allowed[key] = value
    return settings
