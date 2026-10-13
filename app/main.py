import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from . import settings
from .hub import Hub, sse_events
from .model import ExpectedPowerModel
from .replay import load_replay, replay_loop
from .schemas import Prediction, Reading

log = logging.getLogger("app")
state: dict = {}
STATIC = settings.ROOT / "app" / "static"


def _run(readings: list[Reading]) -> list[Prediction]:
    m: ExpectedPowerModel = state["model"]
    preds = m.predict(
        [r.timestamp for r in readings],
        [r.poa_irradiance for r in readings],
        [r.module_temp for r in readings],
        [r.ambient_temp for r in readings],
    )
    return [Prediction(**p) for p in preds]


def predict_event(reading: Reading, source: str) -> dict:
    """One reading -> the event the dashboard displays. `source` is 'replay' now, 'sensor' later."""
    pred = _run([reading])[0]
    return {
        "source": source,
        "sensor_id": reading.sensor_id,
        "timestamp": pred.timestamp,
        "poa_irradiance": reading.poa_irradiance,
        "module_temp": reading.module_temp,
        "ambient_temp": reading.ambient_temp,
        "expected_power_norm": pred.expected_power_norm,
        "daytime": pred.daytime,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = ExpectedPowerModel()   # load once at startup
    state["hub"] = Hub()
    task = None
    if settings.SOURCE != "file":
        log.warning("source=sensor: waiting for sensors to POST /predict (no replay file used)")
    elif settings.REPLAY_CSV.exists():
        try:
            readings = load_replay(settings.REPLAY_CSV, settings.REPLAY_MAX_ROWS)
        except Exception as e:                       # the API must still start; the problem is shown on / and on the dashboard
            readings = []
            state["replay_error"] = str(e)
            log.error("replay file NOT used: %s", e)
        else:
            log.warning("replay: %d readings from %s every %ss", len(readings), settings.REPLAY_CSV, settings.REPLAY_INTERVAL_S)
        if readings:
            task = asyncio.create_task(replay_loop(state["hub"], predict_event, readings,
                                                   settings.REPLAY_INTERVAL_S, settings.REPLAY_LOOP))
    else:
        log.warning("replay: %s not found -> no data source (POST /predict still works)", settings.REPLAY_CSV)
    yield
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    state.clear()


app = FastAPI(title="TAYF Expected Power API", version="1.2.0",
              description="Normalized AC expected power for PVDAQ System 10 (NREL CIS-1), ML-A model (POA, module temp, ambient temp, time).",
              lifespan=lifespan)


@app.get("/")
def health():
    m = state.get("model")
    if m is None:
        raise HTTPException(503, "model not loaded")
    hub: Hub = state["hub"]
    return {"status": "ok", "source": settings.SOURCE, "features": m.features, "timezone": m.tz,
            "replay_error": state.get("replay_error"),
            "readings_received": hub.seq,
            "uptime_s": round(time.time() - hub.started),
            "last_reading_age_s": None if hub.last_at is None else round(time.time() - hub.last_at, 1)}


@app.get("/api/latest")
def latest():
    """The most recent reading + prediction (for scripts / other systems that do not want SSE)."""
    ev = state["hub"].latest
    if ev is None:
        raise HTTPException(404, "no readings yet")
    return ev


@app.get("/api/history")
def history(n: int = 100):
    """The last n readings + predictions (max 300, kept in memory)."""
    n = max(1, min(n, 300))
    return state["hub"].history_since(0)[-n:]


@app.post("/predict", response_model=Prediction)
async def predict(reading: Reading):
    """The ONE endpoint: send a reading, get the expected power back, and it shows up live on /dashboard.
    Today the replay of the data file calls the same code; later the real sensors will POST here."""
    ev = state["hub"].publish(predict_event(reading, "sensor"))
    return Prediction(timestamp=ev["timestamp"], expected_power_norm=ev["expected_power_norm"], daytime=ev["daytime"])


@app.get("/stream")
async def stream(request: Request):
    """Server-Sent Events: every reading + prediction as it happens."""
    try:
        last_id = int(request.headers.get("last-event-id", 0) or 0)
    except ValueError:
        last_id = 0
    return StreamingResponse(
        sse_events(state["hub"], last_id, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    dashboard_file = (STATIC / "dashboard.html").resolve()

    if not dashboard_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard file not found: {dashboard_file}"
        )

    return FileResponse(
        path=dashboard_file,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
