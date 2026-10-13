"""Creates a throwaway model_a.json + config.json (same format as the notebook) just to test the API."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

out = Path(__file__).resolve().parent.parent / "artifacts"
out.mkdir(exist_ok=True)
feats = ["POA_IRRADIANCE", "AMBIENT_TEMP", "MODULE_TEMP_2", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
rng = np.random.default_rng(0)
X = rng.uniform(0, 1, (2000, 7)); X[:, 0] *= 1200; X[:, 1] = X[:, 1] * 40 - 5; X[:, 2] = X[:, 2] * 60 + 5
y = X[:, 0] / 1200 * 0.85
XGBRegressor(n_estimators=50).fit(X, y).save_model(out / "model_a.json")
json.dump({"features": feats, "ac_cap_w": 1120, "timezone": "America/Denver", "poa_min": 50,
           "xgboost_version": "dummy"}, open(out / "config.json", "w"), indent=2)
print("dummy artifacts written to", out)