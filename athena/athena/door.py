"""Receives HC-SR04 door status packets from the ESP32-S3 firmware."""
from __future__ import annotations

import socket
import struct
import threading
import time

PACKET = struct.Struct("<4sIHBBB")
MAGIC = b"DOR1"


def parse_packet(packet: bytes):
    if len(packet) != PACKET.size:
        return None
    magic, sequence, distance_cm, present, sensor_ok, buzzer_active = PACKET.unpack(packet)
    if magic != MAGIC or present not in (0, 1) or sensor_ok not in (0, 1) or buzzer_active not in (0, 1):
        return None
    return {
        "sequence": sequence,
        "distance_cm": distance_cm if distance_cm else None,
        "present": bool(present),
        "sensor_ok": bool(sensor_ok),
        "buzzer_active": bool(buzzer_active),
    }


class DoorReceiver:
    def __init__(self, port: int = 5006):
        self.port = port
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_seen = 0.0
        self._state = {
            "distance_cm": None,
            "present": False,
            "sensor_ok": False,
            "buzzer_active": False,
            "sequence": 0,
        }
        self.error = ""

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="door-udp", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(0.5)
        except OSError as exc:
            self.error = f"cannot listen on UDP {self.port}: {exc}"
            sock.close()
            return
        while not self._stop.is_set():
            try:
                packet, _address = sock.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break
            state = parse_packet(packet)
            if state is None:
                continue
            with self._lock:
                self._state = state
                self._last_seen = time.time()
        sock.close()

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
            last_seen = self._last_seen
        connected = last_seen > 0 and time.time() - last_seen <= 3.5
        state["connected"] = connected
        state["status"] = ("present" if state["present"] else "clear") if connected else "offline"
        state["last_seen"] = last_seen or None
        state["error"] = self.error
        if not connected:
            state["present"] = False
            state["buzzer_active"] = False
        return state
