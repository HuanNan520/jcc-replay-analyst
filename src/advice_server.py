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
    """In-memory broadcast hub · keeps a bounded history · new connections can catch up on the last N items."""

    def __init__(self, history_size: int = 20):
        self._clients: set[WebSocket] = set()
        self._history: deque[dict] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
            snapshot = list(self._history)
        # Push history (so a newly connected UI can show the last few advice items)
        for msg in snapshot:
            try:
                await ws.send_text(json.dumps({"type": "history", "payload": msg}))
            except Exception:
                pass

    async def unsubscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, advice: dict) -> int:
        """Broadcast to all live connections · returns the number of successful pushes."""
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
    # Overlay and server run on the same machine · CORS is not a security concern here · allow all
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
        """Producer HTTP entry point · the tick_loop POSTs here."""
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
        """Consumer WS entry point · the overlay subscribes here."""
        await ws.accept()
        await app.state.broadcaster.subscribe(ws)
        log.info("WS client connected · total=%d", len(app.state.broadcaster._clients))
        try:
            # Heartbeat: the client sends ping periodically · the server replies pong
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text("pong")
                # Other messages are ignored (the service is one-way server -> client by design)
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
