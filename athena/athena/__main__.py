import argparse

from .app import run

p = argparse.ArgumentParser(description="WiFi Radar - local Wi-Fi CSI sensing for ESP32-S3")
p.add_argument("--udp", type=int, default=5005, help="UDP port the board streams to")
p.add_argument("--http", type=int, default=8000, help="local web page port")
p.add_argument("--no-browser", action="store_true")
p.add_argument("--edition", choices=("factory", "user"), default="factory", help="factory enables recording/training; user runs inference only")
p.add_argument("--host", default="127.0.0.1", help="HTTP bind address; use 0.0.0.0 only on a trusted local network")
a = p.parse_args()
run(a.udp, a.http, not a.no_browser, edition=a.edition, host=a.host)
