"""Tests for WiFi Radar edition flags and persistent shared preferences."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from athena.edition import (  # noqa: E402
    DEFAULT_NOTIFICATIONS,
    DEFAULT_OUTPUTS,
    FACTORY,
    USER,
    load_settings,
    normalize_edition,
    save_settings,
    update_preferences,
)


def test_edition_names():
    assert normalize_edition(None) == FACTORY
    assert normalize_edition("factory") == FACTORY
    assert normalize_edition("USER") == USER
    try:
        normalize_edition("production")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown edition must be rejected")


def test_notification_settings_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "models", "settings.json")
        settings = load_settings(path)
        assert settings["notifications"] == DEFAULT_NOTIFICATIONS
        assert settings["outputs"] == DEFAULT_OUTPUTS

        update_preferences(settings, "notifications", {"fall": False, "movement": False})
        update_preferences(settings, "outputs", {"buzzer_enabled": False})
        save_settings(path, settings)

        loaded = load_settings(path)
        assert loaded["notifications"]["fall"] is False
        assert loaded["notifications"]["movement"] is False
        assert loaded["notifications"]["presence"] is True
        assert loaded["outputs"]["buzzer_enabled"] is False


def test_unknown_preferences_are_rejected():
    settings = {"notifications": dict(DEFAULT_NOTIFICATIONS),
                "outputs": dict(DEFAULT_OUTPUTS)}
    for section, values in (
        ("notifications", {"unknown": True}),
        ("notifications", {"fall": "false"}),
        ("outputs", {"buzzer_enabled": 1}),
    ):
        try:
            update_preferences(settings, section, values)
        except ValueError:
            continue
        raise AssertionError(f"invalid values should be rejected: {section} {values}")


if __name__ == "__main__":
    test_edition_names()
    test_notification_settings_round_trip()
    test_unknown_preferences_are_rejected()
    print("ALL EDITION TESTS PASSED")
