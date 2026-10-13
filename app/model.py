"""Loads the XGBoost model + config saved by the notebook and predicts expected power."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .Timeutil import to_utc_index

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


class ExpectedPowerModel:
    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        cfg = json.loads((artifacts_dir / "config.json").read_text())
        self.features: list[str] = cfg["features"]       # order matters
        self.tz: str = cfg["timezone"]
        self.poa_min: float = float(cfg.get("poa_min", 50))
        self.trained_with_xgb: str = cfg.get("xgboost_version", "?")
        self.model = XGBRegressor()
        self.model.load_model(artifacts_dir / "model_a.json")

    def _features(self, timestamps: pd.DatetimeIndex, poa, module_temp, ambient_temp) -> pd.DataFrame:
        # same feature engineering as the notebook (add_time_features), local time
        loc = timestamps.tz_convert(self.tz)
        hour = np.asarray(loc.hour + loc.minute / 60)
        doy = np.asarray(loc.dayofyear)
        df = pd.DataFrame({
            "POA_IRRADIANCE": np.asarray(poa, float),
            "MODULE_TEMP_2": np.asarray(module_temp, float),
            "AMBIENT_TEMP": np.asarray(ambient_temp, float),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        })
        return df[self.features]

    def predict(self, timestamps, poa, module_temp, ambient_temp) -> list[dict]:
        ts = to_utc_index(timestamps, self.tz)
        norm = np.clip(self.model.predict(self._features(ts, poa, module_temp, ambient_temp)), 0, None)
        poa = np.asarray(poa, float)
        out = []
        for i in range(len(ts)):
            daytime = poa[i] >= self.poa_min
            expected_norm = float(norm[i]) if daytime else 0.0
            row = {
                "timestamp": ts[i].tz_convert(self.tz).isoformat(),   # site local time with offset
                "expected_power_norm": round(expected_norm, 5),
                "daytime": bool(daytime),
            }
            out.append(row)
        return out