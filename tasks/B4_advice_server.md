# B4 · WebSocket 建议推送服务

**分配给**：Claude Sonnet 4.6（`claude-sonnet-4-6`）· 常规 FastAPI + WebSocket 活。
**依赖**：无 · 可和 B1 / B3 并行。
**预期工时**：2–3 小时。
**运行时**：Python 服务（WSL 或 Windows 两边跑都行 · 建议 Windows 方便和 overlay 同机器）。
**新产品定位中的角色**：**实时 tick loop 与 UI 层的中间件** · 收 advice · 广播到所有订阅客户端。

---

## 你是谁

你是被派到 `HuanNan520/jcc-replay-analyst` 执行 B4 的 Claude Sonnet 4.6。
新产品形态里 · B2 live_tick 产出 Advice → 通过**你写的这个 WebSocket 服务**广播 → B5 PyQt overlay 订阅显示。

你这层是**纯传输 + 广播** · 不做任何 LLM / 感知工作 · 职责极单一。

## 目标产物

```python
# B2 tick_loop 端（生产者）
import httpx

async with httpx.AsyncClient() as c:
    await c.post(
        "http://localhost:8765/advice",
        json=advice.model_dump(),
    )

# B5 overlay 端（消费者）
import websockets, json
async with websockets.connect("ws://localhost:8765/ws/advice") as ws:
    async for msg in ws:
        advice = json.loads(msg)
        render(advice)
```

## 具体要做

### 1. 新增 `src/advice_server.py`

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
    """内存广播中枢 · 保存一个 bounded history · 新连接能补齐最近 N 条。"""

    def __init__(self, history_size: int = 20):
        self._clients: set[WebSocket] = set()
        self._history: deque[dict] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
            snapshot = list(self._history)
        # 推历史（帮新连接 UI 显示最近几条 advice）
        for msg in snapshot:
            try:
                await ws.send_text(json.dumps({"type": "history", "payload": msg}))
            except Exception:
                pass

    async def unsubscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, advice: dict) -> int:
        """广播给所有活连接 · 返回成功推送数。"""
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
    # overlay 和 server 同机 · CORS 不是安全焦点 · 全放
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
        """生产者 HTTP 入口 · B2 tick_loop 调这里。"""
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
        """消费者 WS 入口 · B5 overlay 订阅这里。"""
        await ws.accept()
        await app.state.broadcaster.subscribe(ws)
        log.info("WS client connected · total=%d", len(app.state.broadcaster._clients))
        try:
            # 心跳：客户端定期发 ping · server 回 pong
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text("pong")
                # 其他消息忽略（服务设计是单向 server → client）
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

### 2. 更新 `requirements-windows.txt`（如果 B1 已创建）或单独加

```
fastapi>=0.115
uvicorn[standard]>=0.32
```

这两个跨平台 · 所以既可以加到主 `requirements.txt`（更方便）· 也可以加到 `requirements-windows.txt`（更干净）。

**推荐**加到**主 requirements.txt** —— 这个服务本身不 Windows-only · WSL 也能跑 · 加主 reqs 合理。

### 3. 单元测试 `tests/test_advice_server.py`

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
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/advice") as ws:
        ws.send_text("ping")
        # 第一条可能是 history · 第二条 pong
        msgs = []
        for _ in range(2):
            try:
                msgs.append(ws.receive_text())
            except Exception:
                break
        assert "pong" in msgs
```

### 4. README 更新（加一小段）

在 "实时 coach 模式 · 数据源" 段之后：

```markdown
### 实时 coach 模式 · 运行

```bash
# 1. 起本地 vLLM（和复盘一样）
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-VL-4B-FP8 \
  --port 8000

# 2. 起 advice 广播服务（任意一端）
python -m src.advice_server --port 8765

# 3. 起 live tick loop（Windows · 吃 OBS 虚拟摄像头 · 见 B2）
# 4. 起 overlay UI（Windows · 订阅 ws://localhost:8765/ws/advice · 见 B5）
```
```

---

## 禁止做的事

- 不要加鉴权 / token / API key —— 本地 127.0.0.1 · 跨机扩展等用户明确要求再加
- 不要引入 redis / rabbitmq / kafka 等外部 broker —— 内存 set 足够
- 不要做持久化（数据库 / 文件） —— deque 够了 · 实时场景历史不值钱
- 不要写 graceful shutdown 的复杂逻辑 —— uvicorn 自带的够
- 不要接 B2 / B3 的业务逻辑 —— 这层纯传输
- 不要用 `anthropic` / `openai` 相关模块（当然也不需要）
- 不要改 `src/llm_analyzer.py` / `src/decision_llm.py`（B3 写的）/ `src/schema.py`

---

## 自验收清单

- [ ] `python -c "from src.advice_server import create_app, AdviceBroadcaster"` 导入无错
- [ ] `pytest tests/test_advice_server.py -v` 全绿 · 至少 7 个测试
- [ ] `python -m src.advice_server --port 8765` 能起来 · `curl http://localhost:8765/health` 返回 `{"ok":true,...}`
- [ ] 开两个终端 · 一个 `websocat ws://localhost:8765/ws/advice`（或等价 Python 客户端） · 另一个 `curl -X POST http://localhost:8765/advice -d '{"kind":"augment","reasoning":"t","confidence":0.9,"ranked":[],"recommendation":"-"}' -H 'content-type: application/json'` · WebSocket 那端收到消息
- [ ] 和原 40 个 pytest 一起跑 · 零回归
- [ ] `git diff --stat` 只含：
  - `src/advice_server.py` (新)
  - `tests/test_advice_server.py` (新)
  - `requirements.txt` (加 fastapi + uvicorn)
  - `README.md` (加一小段)

## 完成后

给用户 ≤ 150 字报告：
- `/health` 响应 JSON 样例
- 并发 5 个 WS 客户端 + 一次 POST · 广播成功数（用 `wscat` 或 Python 脚本自己验）
- `/advice` 接受什么 shape 的 body · 给 B2 的契约确认
- 给 B5 的 subscription 契约：ws URL + message 格式 `{"type":"advice","payload":{...}}` 和 `{"type":"history","payload":{...}}`

不 git commit。

---

## 参考

- FastAPI WebSocket: https://fastapi.tiangolo.com/advanced/websockets/
- TestClient WS: https://fastapi.tiangolo.com/advanced/testing-websockets/
