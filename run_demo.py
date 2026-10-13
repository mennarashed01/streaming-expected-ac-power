#!/usr/bin/env python3
"""
One command to run everything, and keep it running:   python run_demo.py

 - starts the API (uvicorn, SOURCE=sensor -> it only listens to sensors)
 - starts the software sensor that reads data/sensor_data.csv and POSTs every reading to /predict
 - if either process dies it is restarted automatically
 - Ctrl+C stops both

Options:  --csv FILE  --interval SECONDS  --mode loop|once|tail  --port 8000  --noise 0.01  --no-browser
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pick_csv(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for name in ("sensor_data.csv", "sensor_data.example.csv"):
        p = ROOT / "data" / name
        if p.exists():
            if name.endswith("example.csv"):
                print(f"NOTE: data/sensor_data.csv not found -> using the small example file. Copy your CSV to data/sensor_data.csv.")
            return p
    sys.exit("No CSV found. Put your file at data/sensor_data.csv (or pass --csv).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--interval", default="1.0")
    ap.add_argument("--mode", default="loop", choices=["loop", "once", "tail"])
    ap.add_argument("--noise", default="0")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    csv_path = pick_csv(a.csv)
    base = f"http://127.0.0.1:{a.port}"

    api_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", a.port]
    sim_cmd = [sys.executable, str(ROOT / "simulator" / "sensor_sim.py"), "--csv", str(csv_path), "--url", base,
               "--interval", a.interval, "--mode", a.mode, "--noise", a.noise]
    env = {**os.environ, "SOURCE": "sensor"}
    procs = {"api": [api_cmd, env], "sensor": [sim_cmd, os.environ.copy()]}
    running: dict[str, subprocess.Popen] = {}
    stop = threading.Event()

    def start(name):
        cmd, e = procs[name]
        running[name] = subprocess.Popen(cmd, cwd=ROOT, env=e)
        print(f"[run_demo] started {name} (pid {running[name].pid})", flush=True)

    start("api")
    time.sleep(2)
    start("sensor")
    print(f"\n[run_demo] dashboard: {base}/dashboard   |   API docs: {base}/docs   |   Ctrl+C to stop\n", flush=True)
    if not a.no_browser:
        threading.Timer(4, lambda: webbrowser.open(f"{base}/dashboard")).start()
    try:
        while not stop.is_set():
            for name, p in list(running.items()):
                code = p.poll()
                if code is not None and not (name == "sensor" and a.mode == "once" and code == 0):
                    print(f"[run_demo] {name} exited with code {code} -> restarting in 3s", flush=True)
                    time.sleep(3)
                    start(name)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in running.values():
            p.terminate()
        for p in running.values():
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("[run_demo] stopped")


if __name__ == "__main__":
    main()
