"""Tests for the streaming logic (hub / SSE / replay). They do not need the model or FastAPI."""
import asyncio
import json
from pathlib import Path

import pandas as pd

from app.hub import Hub, sse_events
from app.replay import load_replay, replay_loop


def _ev(i):
    return {"source": "replay", "timestamp": f"2022-06-21T0{i}:00:00-06:00", "expected_power_norm": 0.1 * i}


def _parse(chunk):
    for line in chunk.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])


def test_hub_sequence_history_and_boot():
    h = Hub(history=2)
    a, b, c = (h.publish(_ev(i)) for i in (1, 2, 3))
    assert (a["seq"], b["seq"], c["seq"]) == (1, 2, 3)
    assert a["boot"] == h.boot
    assert [e["seq"] for e in h.history_since(0)] == [2, 3]      # keeps only the last `history`
    assert [e["seq"] for e in h.history_since(2)] == [3]


def test_sse_resume_live_and_disconnect():
    async def run():
        h = Hub()
        for i in (1, 2, 3):
            h.publish(_ev(i))
        state = {"stop": False}

        async def disconnected():
            return state["stop"]

        got = []
        gen = sse_events(h, last_id=1, is_disconnected=disconnected, keepalive_s=0.2)
        for _ in range(2):                                   # history after id 1 -> seq 2, 3
            got.append(_parse(await gen.__anext__())["seq"])
        assert got == [2, 3]
        assert (await gen.__anext__()).startswith("retry:")
        h.publish(_ev(4))                                    # live event
        assert _parse(await gen.__anext__())["seq"] == 4
        assert (await gen.__anext__()).startswith(": keep-alive")   # idle -> keep-alive
        state["stop"] = True
        async for _ in gen:
            pass
        assert not h._subs                                   # unsubscribed on disconnect
    asyncio.run(run())


def test_sse_stale_last_id_after_server_restart():
    async def run():
        h = Hub()
        h.publish(_ev(1))

        async def disconnected():
            return False
        gen = sse_events(h, last_id=999, is_disconnected=disconnected)   # id from the previous server run
        assert _parse(await gen.__anext__())["seq"] == 1
        await gen.aclose()
    asyncio.run(run())


def test_load_replay_both_formats_and_bad_rows(tmp_path: Path):
    ts = pd.date_range("2022-06-21 12:00", periods=4, freq="min", tz="UTC")
    mine = tmp_path / "mine.csv"
    pd.DataFrame({"timestamp": [t.isoformat() for t in ts], "poa_irradiance": [500, 510, -0.5, 9999],
                  "module_temp": [30, 31, 32, 33], "ambient_temp": [15, 15, 15, 15]}).to_csv(mine, index=False)
    r = load_replay(mine)
    assert len(r) == 3                                       # POA 9999 is out of range -> skipped
    assert r[2].poa_irradiance == 0.0                        # small negative night value clipped to 0

    pv = tmp_path / "pvdaq.csv"
    pd.DataFrame({"utc_measured_on": ts.strftime("%m/%d/%Y %I:%M:%S %p"), "POA_IRRADIANCE": [500, 510, 520, 530],
                  "MODULE_TEMP_2": [30, 31, 32, 33], "AMBIENT_TEMP": [15, 15, 15, 15]}).to_csv(pv, index=False)
    p = load_replay(pv)
    assert len(p) == 4 and p[0].timestamp == r[0].timestamp  # UTC column kept as UTC


def test_replay_loop_publishes_every_row():
    from app.schemas import Reading
    readings = [Reading(timestamp="2022-06-21T18:00:00Z", poa_irradiance=500 + i, module_temp=30, ambient_temp=15) for i in range(3)]

    async def run():
        h = Hub()
        await replay_loop(h, lambda r, src: {"source": src, "poa_irradiance": r.poa_irradiance}, readings, 0, loop_forever=False)
        return h
    h = asyncio.run(run())
    assert [e["poa_irradiance"] for e in h.history_since(0)] == [500, 501, 502]
