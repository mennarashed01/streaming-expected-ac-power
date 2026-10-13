"""In-memory broadcast hub + Server-Sent-Events formatting. No FastAPI imports (easy to test)."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque


class Hub:
    """Every reading + prediction (from the replay OR from real sensors via POST /predict) goes through publish()."""

    def __init__(self, history: int = 300):
        self.boot = uuid.uuid4().hex[:8]      # changes when the server restarts (lets the page reset itself)
        self.seq = 0
        self.started = time.time()
        self.last_at: float | None = None      # wall-clock time of the last published reading
        self._history = deque(maxlen=history)
        self._subs: set[asyncio.Queue] = set()

    def publish(self, event: dict) -> dict:
        self.seq += 1
        event = {**event, "seq": self.seq, "boot": self.boot}
        self._history.append(event)
        self.last_at = time.time()
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:         # slow client: skip instead of blocking everybody
                pass
        return event

    @property
    def latest(self) -> dict | None:
        return self._history[-1] if self._history else None

    def history_since(self, last_id: int) -> list[dict]:
        return [e for e in self._history if e["seq"] > last_id]

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)


def sse_format(event: dict) -> str:
    return f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"


async def sse_events(hub: Hub, last_id: int, is_disconnected, keepalive_s: float = 15.0):
    """Async generator of SSE text chunks: missed history first, then live events."""
    if last_id > hub.seq:                      # the browser remembers an id from before a server restart
        last_id = 0
    q = hub.subscribe()                        # subscribe BEFORE reading history so nothing is missed
    try:
        sent = last_id
        for e in hub.history_since(last_id):
            sent = e["seq"]
            yield sse_format(e)
        yield "retry: 3000\n\n"
        while not await is_disconnected():
            try:
                e = await asyncio.wait_for(q.get(), timeout=keepalive_s)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if e["seq"] <= sent:               # already sent through the history burst
                continue
            sent = e["seq"]
            yield sse_format(e)
    finally:
        hub.unsubscribe(q)
