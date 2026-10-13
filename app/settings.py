"""Streaming settings (override with environment variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# SOURCE = "file"  -> readings come from REPLAY_CSV (now)
# SOURCE = "sensor" -> no file at all, readings come only from sensors that POST to /predict (later)
SOURCE = os.getenv("SOURCE", "file").lower()

# CSV that is replayed as a fake live sensor feed (columns: timestamp, poa_irradiance, module_temp, ambient_temp)
REPLAY_CSV = Path(os.getenv("REPLAY_CSV", ROOT / "data" / "replay.csv"))
REPLAY_INTERVAL_S = float(os.getenv("REPLAY_INTERVAL_S", "1.0"))          # seconds between two readings
REPLAY_MAX_ROWS = int(os.getenv("REPLAY_MAX_ROWS", "20000"))   # a replay file is meant to be ONE day, not the whole year
REPLAY_LOOP = os.getenv("REPLAY_LOOP", "1").lower() not in ("0", "false", "no")
