# B4 · WebSocket advice push service

**Assigned to**: Claude Sonnet 4.6 (`claude-sonnet-4-6`) · routine FastAPI + WebSocket work.
**Dependencies**: none · can run in parallel with B1 / B3.
**Estimated effort**: 2–3 hours.
**Runtime**: a Python service (runs on either WSL or Windows · Windows recommended for co-locating with the overlay).
**Role in the new product positioning**: the **middleware between the real-time tick loop and the UI layer** · receives advice · broadcasts to all subscribed clients.

---

## Who you are

You are the Claude Sonnet 4.6 dispatched to `HuanNan520/jcc-replay-analyst` to execute B4.
In the new product form · B2 live_tick produces Advice → broadcasts via **this WebSocket service you write** → the B5 PyQt overlay subscribes and displays.

This layer is **pure transport + broadcast** · does no LLM / perception work · its responsibility is extremely single-purpose.

## Target deliverable

```python
# B2 tick_loop side (producer)
import httpx

async with httpx.AsyncClient() as c:
    await c.post(
        "http://localhost:8765/advice",
        json=advice.model_dump(),
    )

# B5 overlay side (consumer)
import websockets, json
async with websockets.connect("ws://localhost:8765/ws/advice") as ws:
    async for msg in ws:
        advice = json.loads(msg)
        render(advice)
```

## What to do

### 1. Add `src/advice_server.py`

```python
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

log = logging.getLogger(__name__)


class AdviceBroadcaster:
    """In-memory broadcast hub · keeps a bounded history · new connections catch up on the latest N."""

    def __init__(self, history_size: int = 20):
        self._clients: set[WebSocket] = set()
        self._history: deque[dict] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
            snapshot = list(self._history)
        # push history (helps a new connection's UI show the latest few advice items)
        for msg in snapshot:
            try:
                await ws.send_text(json.dumps({"type": "history", "payload": msg}))
            except Exception:
                pass

    async def unsubscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, advice: dict) -> int:
        """Broadcasts to all live connections · returns the number of successful pushes."""
        async with self._lock:
            self._history.append(advice)
            targets = list(self._clients)
        sent = 0
        dead: list[WebSocket] = []
        msg = json.dumps({"type": "advice", "payload": advice})
        for ws in targets:
            try:
                await ws.send_text(msg)
                sent += 1
            except Exception as e:
                log.debug("broadcast to client failed · %s", e)
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
        return sent


def create_app(broadcaster: AdviceBroadcaster | None = None) -> FastAPI:
    app = FastAPI(title="jcc-coach advice server", version="0.1")
    # overlay and server are on the same machine · CORS isn't a security focus · allow all
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.broadcaster = broadcaster or AdviceBroadcaster()

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "clients": len(app.state.broadcaster._clients),
            "history": len(app.state.broadcaster._history),
        }

    @app.post("/advice")
    async def post_advice(request: Request):
        """Producer HTTP entry · B2 tick_loop calls here."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "invalid JSON body")
        if not isinstance(body, dict) or "kind" not in body:
            raise HTTPException(400, "body must be dict with at least 'kind'")
        sent = await app.state.broadcaster.broadcast(body)
        return {"broadcast_to": sent}

    @app.websocket("/ws/advice")
    async def ws_advice(ws: WebSocket):
        """Consumer WS entry · the B5 overlay subscribes here."""
        await ws.accept()
        await app.state.broadcaster.subscribe(ws)
        log.info("WS client connected · total=%d", len(app.state.broadcaster._clients))
        try:
            # heartbeat: the client sends ping periodically · the server replies pong
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text("pong")
                # ignore other messages (the service is one-way server → client by design)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.info("WS client error: %s", e)
        finally:
            await app.state.broadcaster.unsubscribe(ws)
            log.info("WS client disconnected · total=%d", len(app.state.broadcaster._clients))

    return app


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
```

### 2. Update `requirements-windows.txt` (if B1 already created it) or add separately

```
fastapi>=0.115
uvicorn[standard]>=0.32
```

These two are cross-platform · so you can either add them to the main `requirements.txt` (more convenient) · or to `requirements-windows.txt` (cleaner).

**Recommended**: add them to the **main requirements.txt** — this service isn't Windows-only · WSL can run it too · adding to the main reqs is reasonable.

### 3. Unit test `tests/test_advice_server.py`

```python
import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from src.advice_server import create_app, AdviceBroadcaster


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["clients"] == 0


def test_post_advice_no_clients():
    app = create_app()
    client = TestClient(app)
    r = client.post("/advice", json={"kind": "augment", "reasoning": "test", "confidence": 0.9})
    assert r.status_code == 200
    assert r.json()["broadcast_to"] == 0


def test_post_advice_rejects_bad_body():
    app = create_app()
    client = TestClient(app)
    r = client.post("/advice", json={"not_kind": "bad"})
    assert r.status_code == 400


def test_post_advice_rejects_non_json():
    app = create_app()
    client = TestClient(app)
    r = client.post("/advice", data="not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_websocket_roundtrip():
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/advice") as ws:
        # send advice
        client.post("/advice", json={"kind": "level", "reasoning": "x", "confidence": 0.5, "action": "up"})
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "advice"
        assert msg["payload"]["kind"] == "level"


def test_websocket_replays_history():
    app = create_app()
    client = TestClient(app)
    # post two first · then connect ws · should receive history
    for i in range(2):
        client.post("/advice", json={"kind": "augment", "reasoning": f"hist{i}", "confidence": 0.7, "ranked": [], "recommendation": "-"})
    with client.websocket_connect("/ws/advice") as ws:
        msgs = []
        for _ in range(2):
            msgs.append(json.loads(ws.receive_text()))
        assert all(m["type"] == "history" for m in msgs)


def test_broadcaster_history_bounded():
    b = AdviceBroadcaster(history_size=3)
    asyncio.run(_push_many(b, 5))
    assert len(b._history) == 3


async def _push_many(b, n):
    for i in range(n):
        await b.broadcast({"kind": "shop", "i": i})


def test_ping_pong():
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/advice") as ws:
        ws.send_text("ping")
        # the first message might be history · the second is pong
        msgs = []
        for _ in range(2):
            try:
                msgs.append(ws.receive_text())
            except Exception:
                break
        assert "pong" in msgs
```

### 4. README update (add a small section)

After the "Real-time coach mode · data source" section:

```markdown
### Real-time coach mode · running

```bash
# 1. start the local vLLM (same as for replay)
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-VL-4B-FP8 \
  --port 8000

# 2. start the advice broadcast service (either side)
python -m src.advice_server --port 8765

# 3. start the live tick loop (Windows · consumes the OBS Virtual Camera · see B2)
# 4. start the overlay UI (Windows · subscribes to ws://localhost:8765/ws/advice · see B5)
```
```

---

## What not to do

- Don't add auth / tokens / API keys — local 127.0.0.1 · add them only when the user explicitly needs cross-machine extension
- Don't introduce external brokers like redis / rabbitmq / kafka — an in-memory set is enough
- Don't add persistence (database / file) — a deque is enough · real-time history isn't worth much
- Don't write complex graceful-shutdown logic — uvicorn's built-in is enough
- Don't wire in B2 / B3 business logic — this layer is pure transport
- Don't use `anthropic` / `openai` modules (you don't need them anyway)
- Don't modify `src/llm_analyzer.py` / `src/decision_llm.py` (B3's) / `src/schema.py`

---

## Self-acceptance checklist

- [ ] `python -c "from src.advice_server import create_app, AdviceBroadcaster"` imports without error
- [ ] `pytest tests/test_advice_server.py -v` all green · at least 7 tests
- [ ] `python -m src.advice_server --port 8765` starts · `curl http://localhost:8765/health` returns `{"ok":true,...}`
- [ ] Two terminals · one `websocat ws://localhost:8765/ws/advice` (or an equivalent Python client) · the other `curl -X POST http://localhost:8765/advice -d '{"kind":"augment","reasoning":"t","confidence":0.9,"ranked":[],"recommendation":"-"}' -H 'content-type: application/json'` · the WebSocket side receives the message
- [ ] Run together with the original 40 pytest · zero regressions
- [ ] `git diff --stat` only contains:
  - `src/advice_server.py` (new)
  - `tests/test_advice_server.py` (new)
  - `requirements.txt` (add fastapi + uvicorn)
  - `README.md` (add a small section)

## After completion

Give the user a ≤ 150-word report:
- A sample of the `/health` response JSON
- 5 concurrent WS clients + one POST · the broadcast success count (verify with `wscat` or your own Python script)
- What shape of body `/advice` accepts · confirm the contract for B2
- The subscription contract for B5: ws URL + message formats `{"type":"advice","payload":{...}}` and `{"type":"history","payload":{...}}`

No git commit.

---

## References

- FastAPI WebSocket: https://fastapi.tiangolo.com/advanced/websockets/
- TestClient WS: https://fastapi.tiangolo.com/advanced/testing-websockets/
