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
        # 发 advice
        client.post("/advice", json={"kind": "level", "reasoning": "x", "confidence": 0.5, "action": "up"})
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "advice"
        assert msg["payload"]["kind"] == "level"


def test_websocket_replays_history():
    app = create_app()
    client = TestClient(app)
    # 先 post 两条 · 再连 ws · 应该收到 history
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
    """No history in fresh app · send ping · expect pong as first message."""
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/advice") as ws:
        ws.send_text("ping")
        # Fresh app has no history · first (and only) reply should be "pong"
        msg = ws.receive_text()
        assert msg == "pong"
