# TAYF Expected Power API (FastAPI)

Predicts the normalized AC expected power (AC power / rated capacity, about 0..1) for PVDAQ System 10 (NREL CIS-1, Golden CO)
and shows it on a live dashboard.

## The idea: one API, two data sources
There is **one** endpoint that makes predictions: `POST /predict`. Only the *source of the readings* changes:

| phase | where the readings come from | setting |
|---|---|---|
| **Now** (no sensors) | a data file (`data/replay.csv`) played one row per second | `SOURCE=file` (default) |
| **Later** (sensors ready) | the sensors send each reading to `POST /predict` | `SOURCE=sensor` |

```
NOW:    data/replay.csv --> (replay) --> model --> dashboard
LATER:  sensors --POST /predict--> model --> dashboard        (the file is not used at all)
```
Switching = set `SOURCE=sensor` (you can also delete `data/replay.csv`). No code change, the dashboard is identical.

## Always-on setup with a software sensor (recommended)

```
data/sensor_data.csv --> [sensor_sim.py] --POST /predict--> [API + model] --SSE--> /dashboard
   (your CSV)            (software sensor)                   (always running)
```
The sensor is a separate process that behaves like real hardware: one reading per second, retries if the API is down.
The API only reacts to what arrives, so later you stop the simulator and point real sensors at the same `POST /predict`.

1. Copy your CSV to `data/sensor_data.csv` (columns `timestamp, poa_irradiance, module_temp, ambient_temp`).
   Timestamps without a timezone are sent as **UTC** by the sensor (`--naive-tz local` to change that).
2. Pick one:
   - **One command, any OS:** `python run_demo.py` (starts API + sensor, restarts either if it dies, opens the dashboard)
   - **Docker, survives crashes and reboots:** `docker compose up -d --build`  then open http://localhost:8000/dashboard
   - **Two terminals:** `SOURCE=sensor uvicorn app.main:app` and `python simulator/sensor_sim.py --csv data/sensor_data.csv`
3. Update the data while it runs: edit/replace `data/sensor_data.csv` (mode `loop` reloads it), or use `--mode tail`
   (`MODE=tail` in docker) and **append rows** - each new row is sent and appears live.

Sensor options: `--interval 1.0`, `--mode loop|once|tail`, `--noise 0.01` (adds sensor-like noise), `--sensor-id NAME`, `--url http://host:8000`.

Extra endpoints: `GET /api/latest`, `GET /api/history?n=100`; `GET /` now also shows `readings_received`, `uptime_s`, `last_reading_age_s`
(handy for monitoring: a growing `last_reading_age_s` means the sensor stopped). `POST /predict` accepts an optional `"sensor_id"`.

## Run (original mode: replay inside the API)
1. `artifacts/` must contain `model_a.json` + `config.json` (saved by the notebook).
2. `pip install -r requirements.txt`  (xgboost is pinned to the training version)
3. Make the data file (ONE clean day of consecutive readings, from the big raw PVDAQ CSV; do not copy the raw file itself):
   `python scripts/pick_clean_row.py --csv "PVDAQ_System10_2020_2023.csv" --day 2022-06-21 --out data/replay.csv`
4. `uvicorn app.main:app --reload`
   - Live dashboard: http://127.0.0.1:8000/dashboard
   - API docs: http://127.0.0.1:8000/docs
5. Docker: `docker build -t pv-api . && docker run -p 8000:8000 pv-api`
6. Tests: `pytest`

`data/replay.csv` columns: `timestamp, poa_irradiance, module_temp, ambient_temp`
(the raw PVDAQ names `utc_measured_on, POA_IRRADIANCE, MODULE_TEMP_2, AMBIENT_TEMP` also work).

Environment variables: `SOURCE` (`file`|`sensor`), `REPLAY_CSV` (default `data/replay.csv`), `REPLAY_INTERVAL_S` (default `1.0`), `REPLAY_LOOP` (default `1`, `0` = play once).

## Endpoints
- `POST /predict` one reading in, expected power out; the reading also appears live on the dashboard
- `GET /dashboard` the live page
- `GET /stream` the live feed the page listens to (Server-Sent Events)
- `GET /` health check (shows the current `source`)

Request:
```json
{"timestamp": "10/11/2010 9:18:00 AM", "poa_irradiance": 800, "module_temp": 35, "ambient_temp": 15}
```
Response:
```json
{"timestamp": "2010-10-11T09:18:00-06:00", "expected_power_norm": 0.585, "daytime": true}
```

## Sensor example (for later)
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"timestamp":"2022-06-21T18:00:00Z","poa_irradiance":900,"module_temp":45,"ambient_temp":25}'
```

## Notes
- `timestamp`: ISO 8601 or text like `10/11/2010 9:18:00 AM` (slash dates are MONTH-first).
  No timezone = site local time (America/Denver). With `Z` or an offset = that timezone. Real sensors should send UTC with `Z`.
- `poa_irradiance` in W/m2 (small negative night values are clipped to 0), temperatures in °C.
- `daytime` = POA >= 50 W/m2. When false, `expected_power_norm` is 0 (the model was trained on daytime data only).
- To get watts: `expected_power_norm * 1120` (AC rating of System 10).
- The dashboard history is kept in memory: it is lost when the server restarts, and a free host that sleeps stops the replay while asleep.
"# streaming-expected-ac-power" 
