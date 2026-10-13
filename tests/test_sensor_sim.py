"""Tests for the software sensor (standard library only). Run: pytest tests/test_sensor_sim.py"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulator"))
import sensor_sim as s  # noqa: E402


def _csv(tmp_path, text):
    p = tmp_path / "d.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_project_columns_and_marks_naive_time_as_utc(tmp_path):
    p = _csv(tmp_path, "timestamp,poa_irradiance,module_temp,ambient_temp\n2020-02-16T16:24:00,755.69,20.7,2.9\n")
    r = s.load_rows(p)
    assert r == [{"timestamp": "2020-02-16T16:24:00Z", "poa_irradiance": 755.69, "module_temp": 20.7, "ambient_temp": 2.9}]


def test_keeps_explicit_timezone_and_local_option(tmp_path):
    p = _csv(tmp_path, "timestamp,poa_irradiance,module_temp,ambient_temp\n2020-02-16T16:24:00+00:00,1,2,3\n2020-02-16 09:00:00,1,2,3\n")
    assert s.load_rows(p)[0]["timestamp"] == "2020-02-16T16:24:00+00:00"
    assert s.load_rows(p, "utc")[1]["timestamp"] == "2020-02-16T09:00:00Z"
    assert s.load_rows(p, "local")[1]["timestamp"] == "2020-02-16 09:00:00"      # API reads it as site local time


def test_raw_pvdaq_column_names(tmp_path):
    p = _csv(tmp_path, "utc_measured_on,POA_IRRADIANCE__421,MODULE_TEMP_2__1,AMBIENT_TEMP__9\n2022-06-21T18:00:00Z,900,45,25\n")
    assert s.load_rows(p)[0]["poa_irradiance"] == 900


def test_bad_rows_are_skipped(tmp_path):
    p = _csv(tmp_path, "timestamp,poa_irradiance,module_temp,ambient_temp\n2020-01-01T10:00:00Z,abc,1,2\n,5,1,2\n2020-01-01T11:00:00Z,nan,1,2\n2020-01-01T12:00:00Z,500,30,20\n2020-01-01T13:00:00Z,500\n")
    assert [r["timestamp"] for r in s.load_rows(p)] == ["2020-01-01T12:00:00Z"]


def test_missing_column_is_a_clear_error(tmp_path):
    p = _csv(tmp_path, "timestamp,poa_irradiance,module_temp\n2020-01-01T12:00:00Z,500,30\n")
    with pytest.raises(ValueError, match="ambient_temp"):
        s.load_rows(p)


def test_noise_stays_inside_api_limits_and_is_off_by_default():
    row = {"timestamp": "x", "poa_irradiance": 1499.0, "module_temp": 99.0, "ambient_temp": -59.0}
    assert s.jitter(row, 0, random.Random(1)) is row
    rng = random.Random(1)
    for _ in range(500):
        o = s.jitter(row, 0.5, rng)
        for k, (lo, hi) in s.LIMITS.items():
            assert lo <= o[k] <= hi
