"""
Pick ONE clean row from the PVDAQ System 10 CSV, using the SAME cleaning rules as the notebook
(section 2 + the solar-elevation filter of section 3), and optionally test it against the running API.

Usage:
    python pick_clean_row.py --csv "PVDAQ_System10_2020_2023.csv"
    python pick_clean_row.py --csv data.csv --seed 7 --n 3
    python pick_clean_row.py --csv data.csv --url http://127.0.0.1:8000/predict
    python pick_clean_row.py --csv data.csv --n 100 --out clean_100_rows.csv     (save 100 clean rows to a CSV)

For a realistic STREAM (consecutive readings of one day, in time order) use --day (site local date):
    python pick_clean_row.py --csv data.csv --day 2022-06-21 --out replay.csv
    python pick_clean_row.py --csv data.csv --day 2022-06-21 --every 5 --out replay.csv     (keep every 5th reading)

In Jupyter / Colab (do NOT run it with plain python inside a cell), use one of:
    !python pick_clean_row.py --csv /content/data.csv --n 100 --out clean_100_rows.csv
or
    from pick_clean_row import main
    main(["--csv", "/content/data.csv", "--n", "100", "--out", "clean_100_rows.csv"])

Needs: pandas, numpy   (pvlib recommended: pip install pvlib  -> enables the solar-elevation filter)
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ==== EDIT THESE when running inside Jupyter / Colab (a CLI argument overrides them) ====
CSV_PATH = ""                        # e.g. "/content/PVDAQ_System10_2020_2023.csv"
N_ROWS = 100                         # how many clean rows to pick
OUT_FILE = "clean_100_rows.csv"      # output CSV (set "" to just print the rows instead)
# ======================================================================================

# ---- same constants as the notebook (section 0 / 0.2) ----------------------------------------
TIME_COL = "utc_measured_on"
CANON = ["utc_measured_on", "AC_POWER", "DC_POWER", "MODULE_TEMP_2", "AMBIENT_TEMP", "POA_IRRADIANCE"]
LAT, LON, ALT = 39.7404, -105.1774, 1792.8
AC_CAP = 1120                       # W  (AC_POWER is in W for System 10)
POA_MIN, POA_MAX = 50, 1500
TEMP_RANGE = (-40, 100)
DOWNTIME_POA = 200
POA_BIN = 50
MAD_K_LOW, MAD_K_HIGH = 3.5, 4.0
ELEV_MIN = 5
TARGET = "AC_POWER_NORM"


def load(path):
    raw = pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path)
    lookup = {c.lower(): c for c in CANON}
    # raw PVDAQ columns carry a sensor-id suffix: "poa_irradiance__421" -> "POA_IRRADIANCE"
    raw.columns = [lookup.get(re.sub(r"__\d+$", "", str(c).strip().lower()), re.sub(r"__\d+$", "", str(c).strip()))
                   for c in raw.columns]
    if TIME_COL in raw.columns:
        raw[TIME_COL] = pd.to_datetime(raw[TIME_COL], utc=True, errors="coerce")
    elif "measured_on" in raw.columns:
        # raw PVDAQ local time (no timezone) -> localize to the site, then convert to UTC like the notebook's column
        local = pd.to_datetime(raw["measured_on"], errors="coerce")
        raw[TIME_COL] = local.dt.tz_localize("America/Denver", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    missing = [c for c in CANON if c not in raw.columns]
    if missing:
        sys.exit(f"Missing expected columns: {missing}  (found: {list(raw.columns)})")
    return raw.dropna(subset=[TIME_COL]).sort_values(TIME_COL).set_index(TIME_COL)


def solar_elevation_noaa(index):
    """Fallback solar elevation (deg, no refraction), NOAA formulas. Accurate to ~0.1 deg: enough for the > 5 deg filter."""
    t = index.tz_convert("UTC")
    doy = np.asarray(t.dayofyear, float)
    hr = np.asarray(t.hour + t.minute / 60 + t.second / 3600, float)
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (hr * 60 + eqtime + 4 * LON) % 1440
    ha = np.radians(tst / 4 - 180)
    lat = np.radians(LAT)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))


def clean(raw):
    """Notebook section 2 (+ elevation filter from section 3). Returns the cleaned frame and a funnel."""
    funnel = []
    log = lambda name, d: funnel.append((name, len(d)))

    df = raw[["POA_IRRADIANCE", "MODULE_TEMP_2", "AMBIENT_TEMP", "AC_POWER"]].copy()
    log("raw", df)

    df = df.dropna(subset=["POA_IRRADIANCE", "AC_POWER"])
    log("drop missing POA / AC_POWER", df)

    df.loc[~df["MODULE_TEMP_2"].between(*TEMP_RANGE), "MODULE_TEMP_2"] = np.nan
    df = df[df["MODULE_TEMP_2"].notna()]
    df = df[df["POA_IRRADIANCE"].between(0, POA_MAX)]
    df = df[df["AC_POWER"].between(0, AC_CAP)]
    log("physical bounds", df)

    df[TARGET] = df["AC_POWER"] / AC_CAP

    df = df[df["POA_IRRADIANCE"] >= POA_MIN]
    log(f"daytime (POA >= {POA_MIN})", df)

    df = df[~((df["POA_IRRADIANCE"] >= DOWNTIME_POA) & (df[TARGET] < 0.01))]
    log("downtime (AC~0 with high POA)", df)

    ratio = df[TARGET] / (df["POA_IRRADIANCE"] / 1000)
    bins = pd.cut(df["POA_IRRADIANCE"], np.arange(0, POA_MAX + POA_BIN, POA_BIN))
    g = ratio.groupby(bins, observed=True)
    med = g.transform("median")
    mad = ((ratio - med).abs().groupby(bins, observed=True).transform("median") * 1.4826).clip(lower=0.02 * med)
    cnt = g.transform("size")
    keep = (cnt < 30) | ((ratio >= med - MAD_K_LOW * mad) & (ratio <= med + MAD_K_HIGH * mad))
    df = df[keep].copy()
    log("robust healthy filter", df)

    # solar elevation > ELEV_MIN (notebook section 3). pvlib if it works, otherwise a NOAA fallback.
    try:
        import pvlib
        sp = pvlib.solarposition.get_solarposition(df.index, LAT, LON, altitude=ALT)
        elev = sp["apparent_elevation"].to_numpy()
        airmass = np.asarray(pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"].to_numpy()), float)
        df = df[(elev > ELEV_MIN) & ~np.isnan(airmass)]
        log(f"solar elevation > {ELEV_MIN} deg (pvlib)", df)
    except Exception as e:   # pvlib missing or failing -> do not crash
        print(f"NOTE: pvlib step skipped ({type(e).__name__}: {e}) -> using the NOAA fallback for the elevation filter\n")
        df = df[solar_elevation_noaa(df.index) > ELEV_MIN]
        log(f"solar elevation > {ELEV_MIN} deg (NOAA fallback)", df)

    # the API requires ambient temperature (the notebook does not filter it, so drop NaN here)
    df = df.dropna(subset=["AMBIENT_TEMP"])
    log("AMBIENT_TEMP present (API needs it)", df)
    return df, funnel


def call_api(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"API error {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach the API at {url}: {e.reason}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=N_ROWS, help="how many clean rows to pick")
    ap.add_argument("--day", help="YYYY-MM-DD (site local date): take ALL clean rows of that day, in time order, instead of a random sample")
    ap.add_argument("--every", type=int, default=1, help="with --day: keep every k-th row (e.g. 5 = one reading per 5 samples)")
    ap.add_argument("--out", default=OUT_FILE or None, help="save the picked rows to this CSV file (e.g. clean_100_rows.csv)")
    ap.add_argument("--url", help="e.g. http://127.0.0.1:8000/predict  (optional: sends the row and compares)")
    a, _unknown = ap.parse_known_args(argv)   # tolerant: Jupyter/Colab inject their own args (-f kernel.json)

    if not a.csv:
        sys.exit('No CSV given. Set CSV_PATH at the top of this file, or pass --csv "path/to/file.csv".')
    df, funnel = clean(load(a.csv))
    print("Cleaning funnel (same as notebook):")
    for name, n in funnel:
        print(f"  {name:42} {n:>10,}")
    if df.empty:
        sys.exit("No clean rows left.")

    if a.day:
        local_day = df.index.tz_convert("America/Denver").strftime("%Y-%m-%d")
        day_df = df[local_day == a.day]
        if day_df.empty:
            days = pd.Series(local_day).value_counts().sort_index()
            sys.exit(f"No clean rows on {a.day}. Days with the most clean rows:\n" + days.sort_values(ascending=False).head(8).to_string())
        picked = day_df.iloc[::max(a.every, 1)]
    else:
        picked = df.sample(min(a.n, len(df)), random_state=a.seed).sort_index()
    if a.out:
        out = pd.DataFrame({
            "timestamp": picked.index.map(lambda x: x.isoformat()),
            "poa_irradiance": picked["POA_IRRADIANCE"].round(4).to_numpy(),
            "module_temp": picked["MODULE_TEMP_2"].round(4).to_numpy(),
            "ambient_temp": picked["AMBIENT_TEMP"].round(4).to_numpy(),
        })
        out.to_csv(a.out, index=False)
        print(f"\nSaved {len(out)} clean rows -> {a.out}")
        return

    for ts, row in picked.iterrows():
        payload = {
            "timestamp": ts.isoformat(),            # UTC, because the CSV column is utc_measured_on
            "poa_irradiance": round(float(row["POA_IRRADIANCE"]), 4),
            "module_temp": round(float(row["MODULE_TEMP_2"]), 4),
            "ambient_temp": round(float(row["AMBIENT_TEMP"]), 4),
        }
        actual_norm = float(row[TARGET])
        print("\n--- request (paste into /docs or curl) ---")
        print(json.dumps(payload, indent=2))
        print(f"actual AC_POWER = {row['AC_POWER']:.1f} W  ->  actual normalized = {actual_norm:.4f}")

        if a.url:
            out = call_api(a.url, payload)
            pred = out["expected_power_norm"]
            err = pred - actual_norm
            print(f"API expected_power_norm = {pred:.4f}  (~{pred * AC_CAP:.0f} W)")
            print(f"error = {err * 100:+.2f} % of rated  ({err * AC_CAP:+.0f} W)")


if __name__ == "__main__":
    main()
