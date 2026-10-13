"""Timestamp handling (no xgboost dependency, so it can be tested alone)."""
from datetime import datetime

import pandas as pd


def parse_timestamp(v):
    """Accepts ISO 8601 (with/without offset or 'Z') and common text formats such as
    '10/11/2010 9:18:00 AM', '10/11/2010 09:18', '2010-10-11 09:18:00'.
    Slash dates are read MONTH-first (US PVDAQ CSV style). Returns a datetime (naive or aware)."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:
        ts = pd.to_datetime(s, format="mixed", dayfirst=False)
    except (ValueError, TypeError):
        return v  # let pydantic raise a clear 422
    return v if pd.isna(ts) else ts.to_pydatetime()


def to_utc_index(timestamps, site_tz: str) -> pd.DatetimeIndex:
    """Naive timestamps are site LOCAL time (site_tz); aware ones keep their own offset."""
    out = []
    for t in timestamps:
        t = pd.Timestamp(t)
        if t.tzinfo is None:
            t = t.tz_localize(site_tz, ambiguous=True, nonexistent="shift_forward")  # ambiguous -> DST side
        out.append(t.tz_convert("UTC"))
    return pd.DatetimeIndex(out)