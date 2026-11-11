"""Four-class Factory pipeline and door telemetry contract checks."""
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from athena import app, data  # noqa: E402
from athena.door import parse_packet  # noqa: E402
import fake_esp  # noqa: E402


def test_door_packet():
    packet = struct.pack("<4sIHBBB", b"DOR1", 17, 84, 1, 1, 1)
    state = parse_packet(packet)
    assert state is not None
    assert state["sequence"] == 17
    assert state["distance_cm"] == 84
    assert state["present"] is True
    assert state["sensor_ok"] is True
    assert state["buzzer_active"] is True
    assert parse_packet(b"invalid") is None
    print("HC-SR04 telemetry packet ok")


def test_four_class_dataset():
    assert data.ACTIVITY_LABELS == ("empty_room", "person_static", "movement", "fall_detection")
    with tempfile.TemporaryDirectory() as root:
        base = 60 + 25 * np.sin(np.linspace(0, 6, 64))
        ts = np.arange(0, 6000, 20, dtype=np.uint32)
        for label_index, label in enumerate(data.ACTIVITY_LABELS):
            for take in range(3):
                rng = np.random.default_rng(label_index * 100 + take)
                amp = np.stack([
                    fake_esp.frame(float(t) / 1000.0, label, base, rng) for t in ts
                ]).clip(0, 255).astype(np.uint8)
                data.save_take(
                    root, label, f"person_{take}", f"room_{take}",
                    ts, np.full(len(ts), -50, dtype=np.int8), amp,
                )

        takes = data.list_takes(root)
        assert set(takes) == set(data.ACTIVITY_LABELS)
        assert all(len(takes[label]) == 3 for label in data.ACTIVITY_LABELS)
        labels, splits, counts = data.load_dataset(root, fs=50, win=64, hop=16)
        assert labels == list(data.ACTIVITY_LABELS)
        assert all(counts[part] == 4 for part in ("train", "val", "test"))
        for X, y in splits.values():
            assert X.shape[1:] == (64, 64)
            assert len(X) == len(y) > 0
        print("four-class dataset split ok:", labels, counts)

        try:
            data.save_take(
                root, "walking", "P1", "R1", ts,
                np.full(len(ts), -50, dtype=np.int8),
                np.zeros((len(ts), 64), dtype=np.uint8),
            )
            raise AssertionError("extra labels must be rejected")
        except ValueError:
            pass

        data.delete_label(root, "movement")
        try:
            data.load_dataset(root)
            raise AssertionError("training must require all four classes")
        except data.DatasetError as exc:
            assert "movement" in str(exc)
        print("unknown and missing class guards ok")


def test_edition_guards():
    with tempfile.TemporaryDirectory() as root:
        old_data, old_models = app.DATA_DIR, app.MODEL_DIR
        app.DATA_DIR = os.path.join(root, "data")
        app.MODEL_DIR = os.path.join(root, "models")
        app.MODEL_PATH = os.path.join(app.MODEL_DIR, "activity_model.pt")
        try:
            user = app.State(5005, edition="user")
            assert "Factory Edition" in user.start_record("movement", "P1", "R1", 20)
            assert "Factory Edition" in user.start_train(5)
            snapshot = user.snapshot()
            assert snapshot["takes"] == {}
            assert snapshot["rec"]["status"] == "disabled"
            assert snapshot["train"]["status"] == "disabled"
            assert snapshot["door"]["connected"] is False

            factory = app.State(5005, edition="factory")
            factory.rx.connected = lambda: True
            message = factory.start_record("running", "P1", "R1", 20)
            assert "الأربع" in message
            print("Factory/User guards ok")
        finally:
            app.DATA_DIR, app.MODEL_DIR = old_data, old_models
            app.MODEL_PATH = os.path.join(old_models, "activity_model.pt")


if __name__ == "__main__":
    test_door_packet()
    test_four_class_dataset()
    test_edition_guards()
    print("ALL FOUR-CLASS PIPELINE TESTS PASSED")
