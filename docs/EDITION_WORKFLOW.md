# WiFi Radar — Editions and release workflow

## Project editions

- **Factory Edition** is used by the research team to receive CSI, record labeled sessions, train models, review metrics, and export a user package.
- **User Edition** runs inference only. Factory recording and training endpoints are disabled. It ships with a trained model and does not include factory recordings or training tools.

Both editions use the same signal format, feature pipeline, model architecture, and inference code. This prevents a model from being trained with one preprocessing path and deployed with another.

## Model classes required for a user release

The release packager requires exactly these four activity classes:
1. `empty_room` — nobody is in the room.
2. `person_static` — a person standing still.
3. `sitting` — a person sitting still.
4. `movement` — a person moving.

Fall-event heuristics remain separate and are not a fifth trained class. The Factory UI and API reject any other activity label. Three independent recordings per class are a technical minimum only; use more people, rooms, sessions, and days for research evaluation.

A package is not created if any class or independent test metrics are missing. This is a structural release gate, not proof of accuracy. Review per-class precision, recall, F1, confusion matrix, false alerts, and held-out people and rooms before approving release.

## Factory to user release

1. Run Factory Edition and collect new, labeled recordings.
2. Ensure every class has enough independent recordings. Three per class is only a technical minimum; more people, rooms, sessions, and days are strongly recommended.
3. Train and review the untouched test set.
4. Check held-out-person and held-out-room results separately.
5. Run the release package command from the repository root:

   `python -m athena.package_user --version 1.0.0 --researcher-approved`

6. Distribute `dist/WiFi-Radar-User.zip` to the target local processing device.
7. The target device runs the same inference pipeline with the bundled model; no development laptop is required during normal use.

## Processing hardware note

The current CSI activity model uses PyTorch. The ESP32-S3 is a CSI collector/transmitter, not assumed to be powerful enough to run the current CNN/BiLSTM model. The user package therefore targets a dedicated local processing device (for example, a Raspberry Pi-class computer or a mini PC). The exact hardware must be selected only after measuring model memory use, CPU inference latency, temperature, and sustained operation on the actual checkpoint.

## Calibration

Room calibration creates a room-specific empty-room baseline. It does not retrain the factory model. The user edition defaults to a 10-minute baseline collection, then checks the frame count and signal continuity before saving the profile. A timeout alone is not considered successful calibration.

## Notifications

Notification preferences are stored centrally on the local processing device so the web UI and Android client can share the same choices. Suppressing a notification never suppresses detection or event logging. Remote push delivery still requires a reachable network and an actual delivery service; the base Android notifier can poll the local API while the phone is on the same network.

## Research and safety boundary

This is a research prototype, not a medical or emergency safety device. Fall alerts must be described as possible events until independent validation establishes real-world performance. Never create fall recordings by risking an actual fall; use supervised, controlled simulations on safe surfaces.
