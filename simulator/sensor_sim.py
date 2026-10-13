#!/usr/bin/env python3
"""
Software sensor (PV weather-station simulator).

Reads a CSV of recorded readings and behaves like a real sensor: every INTERVAL seconds it POSTs ONE reading
to the API (POST /predict). The API predicts the expected power and the dashboard updates live.
Uses the Python standard library only (no pip install needed).

CSV columns (names are case-insensitive, raw PVDAQ names also work):
    timestamp, poa_irradiance, module_temp, ambient_temp

Modes
    loop  (default)  play the file again and again, forever
    once             play the file one time and stop
    tail             play the file once, then keep watching it: rows APPENDED later are sent as they arrive

Examples
    python simulator/sensor_sim.py --csv data/sensor_data.csv
    python simulator/sensor_sim.py --csv data/sensor_data.csv --interval 0.5 --noise 0.01
    python simulator/sensor_sim.py --csv data/sensor_data.csv --mode tail          # append rows to the file -> live
    python simulator/sensor_sim.py --url http://my-server:8000

Every option can also be set with an environment variable (API_URL, SENSOR_CSV, INTERVAL, MODE, NOISE,
NAIVE_TZ, SENSOR_ID, SEED) which is what docker-compose does.

The API keeps running by itself. If the API is down/restarting, this script waits and retries (it never gives up).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ALIASES = {
    "timestamp": "timestamp", "utc_measured_on": "timestamp", "measured_on": "timestamp",
    "poa_irradiance": "poa_irradiance",
    "module_temp": "module_temp", "module_temp_2": "module_temp",
    "ambient_temp": "ambient_temp",
}
FIELDS = ("poa_irradiance", "module_temp", "ambient_temp")
LIMITS = {"poa_irradiance": (0.0, 1500.0), "module_temp": (-40.0, 100.0), "ambient_temp": (-60.0, 60.0)}  # = API validation
HAS_TZ = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def norm_col(c: str) -> str:
    c = re.sub(r"__\d+$", "", str(c).strip().lower().lstrip("\ufeff"))   # 'POA_IRRADIANCE__421' -> 'poa_irradiance'
    return ALIASES.get(c, c)


def load_rows(path, naive_tz: str = "utc") -> list[dict]:
    """CSV -> list of {timestamp, poa_irradiance, module_temp, ambient_temp}. Bad rows are skipped.
    naive_tz='utc': timestamps without a timezone are sent as UTC (what a real sensor does).
    naive_tz='local': sent as-is (the API then reads them as site local time, America/Denver)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = [norm_col(c) for c in next(reader)]
        except StopIteration:
            return rows
        missing = [c for c in ("timestamp", *FIELDS) if c not in header]
        if missing:
            raise ValueError(f"CSV is missing columns {missing} (found {header})")
        idx = {c: header.index(c) for c in ("timestamp", *FIELDS)}
        for rec in reader:
            try:
                ts = rec[idx["timestamp"]].strip()
                vals = {k: float(rec[idx[k]]) for k in FIELDS}
            except (ValueError, IndexError):
                continue                                   # blank / partial / non-numeric row
            if not ts or any(math.isnan(v) for v in vals.values()):
                continue
            if naive_tz == "utc" and not HAS_TZ.search(ts):
                ts = ts.replace(" ", "T") + "Z"
            rows.append({"timestamp": ts, **vals})
    return rows


def jitter(row: dict, noise: float, rng: random.Random) -> dict:
    """Small sensor noise: irradiance +-noise (relative), temperatures +-noise*10 degC. Clamped to the API limits."""
    if noise <= 0:
        return row
    out = dict(row)
    for k in FIELDS:
        v = row[k] * (1 + rng.gauss(0, noise)) if k == "poa_irradiance" else row[k] + rng.gauss(0, noise * 10)
        lo, hi = LIMITS[k]
        out[k] = round(min(hi, max(lo, v)), 4)
    return out


def _request(url: str, payload=None, timeout: float = 10.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def wait_for_api(base: str, log) -> None:
    delay = 1.0
    while True:
        try:
            _request(base + "/", timeout=5)
            log(f"API is up at {base}")
            return
        except Exception as e:
            log(f"waiting for API ({type(e).__name__}) - retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 1.7, 15.0)


def send(base: str, row: dict, sensor_id: str, log) -> dict | None:
    """POST one reading. Network errors -> wait and retry the SAME reading. 4xx -> skip the reading."""
    delay = 1.0
    payload = {**row, "sensor_id": sensor_id}
    while True:
        try:
            return _request(base + "/predict", payload)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                log(f"API rejected a reading (HTTP {e.code}): {e.read().decode()[:200]} -> skipped")
                return None
            log(f"API error HTTP {e.code} - retry in {delay:.0f}s")
        except Exception as e:                              # connection refused / timeout / API restarting
            log(f"cannot reach API ({type(e).__name__}) - retry in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 1.7, 15.0)


def mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def run(a, log=None) -> int:
    log = log or (lambda m: print(f"{time.strftime('%H:%M:%S')}  {m}", flush=True))
    base = a.url.rstrip("/")
    rng = random.Random(a.seed)
    path = Path(a.csv)
    if not path.exists():
        log(f"CSV not found: {path}")
        return 2
    rows = load_rows(path, a.naive_tz)
    if not rows and a.mode != "tail":
        log(f"no valid rows in {path}")
        return 2
    log(f"{len(rows)} readings from {path} | mode={a.mode} | every {a.interval}s | sensor_id={a.sensor_id}")
    wait_for_api(base, log)

    sent_total, last_m, i = 0, mtime(path), 0
    while True:
        if i >= len(rows):                                   # reached the end of what we have
            if a.mode == "once":
                log(f"done: {sent_total} readings sent")
                return 0
            if a.mode == "loop":
                i = 0
                log("end of file -> starting again")
            else:                                            # tail: wait for appended rows
                time.sleep(max(a.interval, 0.5))
                m = mtime(path)
                if m != last_m:
                    last_m = m
                    fresh = load_rows(path, a.naive_tz)
                    if len(fresh) < len(rows):               # file was replaced by a shorter one -> start over
                        i = 0
                    if len(fresh) != len(rows):
                        log(f"file changed: {len(fresh)} rows now (was {len(rows)})")
                    rows = fresh
                continue
        elif a.mode == "loop" and mtime(path) != last_m:     # edited while looping -> pick up the new data
            last_m = mtime(path)
            fresh = load_rows(path, a.naive_tz)
            if fresh:
                rows, i = fresh, 0
                log(f"file changed -> reloaded {len(rows)} readings")
        row = rows[i]
        i += 1
        out = send(base, jitter(row, a.noise, rng), a.sensor_id, log)
        sent_total += 1
        if out and (sent_total <= 3 or sent_total % a.log_every == 0):
            log(f"#{sent_total} {row['timestamp']} POA={row['poa_irradiance']:.0f} "
                f"-> expected {out['expected_power_norm'] * 100:.1f}% of rated")
        time.sleep(a.interval)


def parse_args(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(description="Software PV sensor: POSTs CSV readings to the API.")
    ap.add_argument("--csv", default=env("SENSOR_CSV", "data/sensor_data.csv"))
    ap.add_argument("--url", default=env("API_URL", "http://127.0.0.1:8000"), help="API base URL")
    ap.add_argument("--interval", type=float, default=float(env("INTERVAL", "1.0")), help="seconds between readings")
    ap.add_argument("--mode", choices=["loop", "once", "tail"], default=env("MODE", "loop"))
    ap.add_argument("--noise", type=float, default=float(env("NOISE", "0")), help="sensor noise, e.g. 0.01 = ~1%%")
    ap.add_argument("--naive-tz", choices=["utc", "local"], default=env("NAIVE_TZ", "utc"),
                    help="how to read timestamps that have no timezone (default utc)")
    ap.add_argument("--sensor-id", default=env("SENSOR_ID", "SIM-PV10-01"))
    ap.add_argument("--seed", type=int, default=int(env("SEED", "42")))
    ap.add_argument("--log-every", type=int, default=30, help="print one line every N readings")
    return ap.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except KeyboardInterrupt:
        print("\nsensor stopped")
