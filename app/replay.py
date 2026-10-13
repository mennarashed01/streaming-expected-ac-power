"""Fake 'live sensors': replays a CSV of real readings one row at a time. No FastAPI/xgboost imports."""
import asyncio
import logging
import re
from pathlib import Path

import pandas as pd

from .schemas import Reading

log = logging.getLogger("replay")

# accepted column names (the project's snake_case names OR the notebook / PVDAQ names)
ALIASES = {
    "timestamp": "timestamp", "utc_measured_on": "utc_measured_on",
    "poa_irradiance": "poa_irradiance",
    "module_temp": "module_temp", "module_temp_2": "module_temp",
    "ambient_temp": "ambient_temp",
}


def _norm(c) -> str:
    """'POA_IRRADIANCE__421' -> 'poa_irradiance' (raw PVDAQ columns carry a sensor-id suffix)."""
    c = re.sub(r"__\d+$", "", str(c).strip().lower())
    return ALIASES.get(c, c)


def load_replay(path: Path, max_rows: int = 20000) -> list[Reading]:
    df = pd.read_csv(path)
    df.columns = [_norm(c) for c in df.columns]
    if "measured_on" in df.columns and "timestamp" not in df.columns and "utc_measured_on" not in df.columns:
        df["timestamp"] = df["measured_on"]      # raw PVDAQ local time: no timezone -> the API reads it as site local time
    if "utc_measured_on" in df.columns:        # PVDAQ UTC column -> make the timezone explicit
        ts = pd.to_datetime(df["utc_measured_on"], utc=True, errors="coerce")
        df["timestamp"] = ts.map(lambda x: x.isoformat() if pd.notna(x) else None)
    need = ["timestamp", "poa_irradiance", "module_temp", "ambient_temp"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"replay CSV is missing columns: {missing} (found: {list(df.columns)})")

    if len(df) > max_rows:
        raise ValueError(
            f"replay CSV has {len(df):,} rows (max {max_rows:,}). A replay file should be ONE day of clean data. "
            f"Make it with: python scripts/pick_clean_row.py --csv <raw file> --day YYYY-MM-DD --out data/replay.csv")

    readings, bad = [], 0
    for rec in df[need].dropna().to_dict("records"):
        try:
            readings.append(Reading(**rec))
        except Exception:
            bad += 1
    if bad:
        log.warning("replay: skipped %d invalid rows", bad)
    return readings


async def replay_loop(hub, predict_one, readings: list[Reading], interval_s: float, loop_forever: bool):
    """predict_one(reading) -> event dict (without seq/boot). Publishes one reading every interval_s."""
    while True:
        for r in readings:
            hub.publish(predict_one(r, "replay"))
            await asyncio.sleep(interval_s)
        if not loop_forever:
            return
