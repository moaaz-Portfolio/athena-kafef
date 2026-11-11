"""Synthetic CSI sender for offline pipeline tests and UI smoke checks."""
import argparse
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from athena import proto  # noqa: E402

# Schema-test patterns only; these values are not physical fall recordings or accuracy evidence.
ACTIVITIES = {
    "empty_room": (0.0, 0.3),
    "person_static": (0.08, 1.5),
    "movement": (1.2, 5.0),
    "fall_detection": (0.65, 9.0),
}


def frame(t, label, base, rng):
    frequency, amplitude = ACTIVITIES[label]
    phase = np.linspace(0, 3, 64)
    motion = amplitude * np.sin(2 * np.pi * frequency * t + phase) if frequency else 0.0
    return np.clip(base + motion + rng.normal(0, 0.8, 64), 0, 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--rate", type=float, default=90)
    parser.add_argument("--hold", type=float, default=25)
    args = parser.parse_args()
    rng = np.random.default_rng(1)
    base = 60 + 25 * np.sin(np.linspace(0, 6, 64))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    labels = list(ACTIVITIES)
    started, sequence = time.time(), 0
    print("Fake CSI schema stream: four approved labels; Ctrl+C to stop.")
    while True:
        elapsed = time.time() - started
        label = labels[int(elapsed // args.hold) % len(labels)]
        if sequence % int(args.rate * 5) == 0:
            print(f"  now simulating: {label}")
        if rng.random() > 0.03:
            packet = proto.build(sequence, int(elapsed * 1000), -50,
                                 frame(elapsed, label, base, rng))
            sock.sendto(packet, (args.host, args.port))
        sequence += 1
        time.sleep(1.0 / args.rate * rng.uniform(0.7, 1.3))


if __name__ == "__main__":
    main()
