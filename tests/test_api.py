"""Run: pytest   (needs artifacts/model_a.json + config.json; use scripts/make_dummy_artifacts.py if you have none)"""
from fastapi.testclient import TestClient

from app.main import app

BASE = {"poa_irradiance": 800, "module_temp": 35, "ambient_temp": 15}


def test_health_and_predict():
    with TestClient(app) as c:
        assert c.get("/").json()["status"] == "ok"
        r = c.post("/predict", json={**BASE, "timestamp": "2022-06-21T18:00:00Z"})
        assert r.status_code == 200
        j = r.json()
        assert set(j) == {"timestamp", "expected_power_norm", "daytime"}
        assert 0 <= j["expected_power_norm"] <= 1.5 and j["daytime"] is True


def test_night_is_zero():
    with TestClient(app) as c:
        j = c.post("/predict", json={"timestamp": "2022-06-21T06:00:00Z", "poa_irradiance": -0.55,
                                     "module_temp": 10, "ambient_temp": 5}).json()
        assert j["expected_power_norm"] == 0 and j["daytime"] is False


def test_validation_errors():
    with TestClient(app) as c:
        assert c.post("/predict", json={**BASE, "timestamp": "x"}).status_code == 422
        assert c.post("/predict", json={**BASE, "timestamp": "2022-06-21T18:00:00Z", "poa_irradiance": -500}).status_code == 422
        assert c.post("/predict", json={"timestamp": "2022-06-21T18:00:00Z", "poa_irradiance": 800, "module_temp": 35}).status_code == 422


def test_local_vs_utc_timestamp():
    with TestClient(app) as c:
        local = c.post("/predict", json={**BASE, "timestamp": "10/11/2010 9:18:00 AM"}).json()
        utc = c.post("/predict", json={**BASE, "timestamp": "2010-10-11T15:18:00Z"}).json()
        assert local["expected_power_norm"] == utc["expected_power_norm"]
        assert local["timestamp"].endswith("-06:00")


def test_predict_feeds_dashboard_and_health():
    with TestClient(app) as c:
        r = c.post("/predict", json={**BASE, "timestamp": "2022-06-21T18:00:00Z"})
        assert r.status_code == 200 and set(r.json()) == {"timestamp", "expected_power_norm", "daytime"}
        page = c.get("/dashboard")
        assert page.status_code == 200 and "Expected Power" in page.text
        assert c.get("/").json()["source"] in ("file", "sensor")
