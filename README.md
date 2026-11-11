# WiFi Radar

WiFi Radar is a research prototype for studying indoor presence and activity sensing using Wi-Fi Channel State Information (CSI). The CSI sensing path does not use a camera. This repository contains ESP32-S3 firmware, a Python CSI processing application, an Android notification app, and a factory-to-user model release workflow.

> Research boundary: presence and fall detection are experimental estimates. This is not a medical device or an emergency/life-safety system. Do not use it as a substitute for direct supervision.

## System architecture

1. ESP32-S3 CSI node — collects CSI frames and streams 64-subcarrier amplitude packets to a local processing computer over UDP (default port 5005).
2. Local processing unit — runs the Python application, room baseline, trained activity model, event log, and local HTTP API (default port 8000).
3. Web dashboard — displays room presence, movement, possible fall events, CSI quality, room calibration, notification preferences, and recent events.
4. Android app — polls the local processing API and raises Android notifications according to preferences stored on the processing unit.
5. Door sensing — ESP32-S3 firmware supports an HC-SR04 on GPIO4/GPIO5 and an active buzzer through a transistor driver on GPIO6. Door telemetry is sent to the local Python processor over UDP port 5006 when the CSI stream port is 5005. One HC-SR04 detects presence in its field of view; it cannot reliably determine entry versus exit.

The CSI node and door node are separate systems. Their data can be presented together in WiFi Radar, but their sensing logic remains independent.

## Factory Edition and User Edition

### Factory Edition

Factory Edition is the researcher's workstation mode. It enables CSI recording with activity/person/room metadata, dataset review, class management, model training, test metrics, cross-room and cross-person evaluation, and model release preparation.

Launch from the athena directory:

    pip install -r requirements.txt
    python -m athena --edition factory

### User Edition

User Edition is the inference-only runtime. It loads a model trained and evaluated by the factory, disables factory recording and training endpoints, requires an empty-room calibration (10 minutes by default), saves a room-specific baseline separately from the trained model, and keeps local detection and event history on the processing unit.

A user package can be created only after the checkpoint contains exactly four classes (empty room, person static, movement, and fall detection), test metrics are present, and a researcher explicitly approves release. Rule-based fall-event heuristics remain an additional experimental cross-check:

    python -m athena.package_user --version 1.0.0 --researcher-approved

This command creates dist/WiFi-Radar-User.zip. It excludes factory recordings and training code. It does not certify model accuracy: review per-class precision/recall/F1, confusion matrix, false alerts, and tests on held-out people and rooms before release.

Run the packaged User Edition on a dedicated local processing computer:

    python -m athena --edition user --host 0.0.0.0

The ESP32-S3 streams CSI to that computer. The processing hardware must be selected using measured model memory use, CPU inference latency, temperature, and sustained operation. The current PyTorch CNN/BiLSTM model is not assumed to run on the ESP32-S3 itself. A Raspberry Pi-class device or mini PC may be needed. A laptop is not required for the end user's normal operation when the dedicated processing unit is installed.

## Local networking and Android

- Keep the processing computer and phone on the same trusted local network for local Android notifications.
- Open the dashboard on the processing computer, copy its local API token, and paste it into the Android app together with the processing computer URL.
- The dashboard binds to 127.0.0.1 by default. Use --host 0.0.0.0 only on a trusted private LAN when the phone must connect.
- Do not expose the processing API to the public internet or configure router port forwarding.
- Local detection continues without internet. Notifications to a phone outside the local network require internet and a remote push delivery service; the current Android polling service is LAN-based.

## Room calibration

User Edition collects a 10-minute empty-room baseline. Calibration checks that sufficient CSI frames arrived and rejects major baseline shifts between time segments. Calibration is not model training. Recalibrate after changing the CSI node, router, or room layout. These checks are conservative engineering gates and still require validation in target rooms.

## Firmware installation

The ESP32-S3 firmware is built by GitHub Actions in .github/workflows/bulid.yml. The workflow publishes the ESP Web Tools USB installer through GitHub Pages and uploads the merged first-install image plus the OTA update binary as an Actions artifact. First installation uses Chrome/Edge with USB; later updates can be uploaded from the device dashboard. Build locally with ESP-IDF 5.5+:

    idf.py set-target esp32s3
    idf.py build flash monitor

The firmware exposes its own device API (including /api/status, /api/config, and /api/ota). This firmware API is distinct from the Python processing API (/api/state, /api/config, and /api/events).

## Research data

The previous exploratory recordings and incompatible pretrained checkpoint have been removed so the Factory dataset starts clean. Collect only the four approved labels across multiple independent recordings, people, rooms, and sessions. Person static means a person is present but stationary; empty room means nobody is present. Fall-detection data must use safe, controlled simulations and never real falls.

See:
- HC-SR04 and buzzer wiring: docs/HARDWARE_HC-SR04_BUZZER_AR.md
- Factory/user release workflow: docs/EDITION_WORKFLOW.md
- CSI experiments and evaluation: athena/EXPERIMENTS.md
- Python application guide: athena/README.md
- Android app guide: android-notifier/README_AR.md

## License

Apache-2.0. Uses Espressif ESP-CSI components; components/dns_server is from ESP-IDF examples (Unlicense/CC0).
