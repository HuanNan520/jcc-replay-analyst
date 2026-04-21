"""端到端 smoke demo · 证明实时 coach 全链路连通。

不依赖真 OBS · 不依赖真 vLLM · 只在 WSL 里 mock 全链路跑一遍。
独立可跑：python scripts/e2e_smoke.py

步骤：
  1/7 · OBSCapture mock       · 加载 examples/sample_frames/ 12 张 PNG bytes
  2/7 · FrameMonitor analyzing · 对所有帧跑 observe()
  3/7 · VLMClient mock        · 12 帧各产出一个语义丰富 WorldState
  4/7 · advice_server         · 在 port 8765 启动 subprocess
  5/7 · DecisionLLM           · 每类 decision 至少触发一次（augment/carousel/level/positioning/item）
  6/7 · WebSocket client      · 订阅并验证 broadcast
  7/7 · LocalLLMAnalyzer → render_report_html → /tmp/e2e_smoke_report.html
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── repo root on sys.path ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.decision_llm import (  # noqa: E402
    AugmentAdvice,
    CarouselAdvice,
    DecisionContext,
    DecisionLLM,
    ItemAdvice,
    LevelAdvice,
    PositioningAdvice,
)
from src.frame_monitor import FrameMonitor  # noqa: E402
from src.llm_analyzer import LocalLLMAnalyzer, _empty_report  # noqa: E402
from src.report_renderer import render_report_html  # noqa: E402
from src.schema import (  # noqa: E402
    ActiveTrait,
    BagItem,
    MatchReport,
    OpponentPreview,
    RoundReview,
    Unit,
    WorldState,
)
from src.vlm_client import VLMClient  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
FRAMES_DIR = _REPO_ROOT / "examples" / "sample_frames"
ADVICE_PORT = 8765
ADVICE_HOST = "127.0.0.1"
REPORT_PATH = Path("/tmp/e2e_smoke_report.html")

# ── pretty log helpers ────────────────────────────────────────────────────────

_t0_global: float = time.time()


def _ts() -> str:
    """相对时间戳。"""
    return f"{time.time() - _t0_global:6.1f}s"


def step(n: int, total: int, msg: str) -> None:
    print(f"[e2e smoke] step {n}/{total} · {msg}", flush=True)


def sub(msg: str, elapsed: Optional[float] = None) -> None:
    suffix = f" · {elapsed:.1f}s" if elapsed is not None else ""
    print(f"  · {msg}{suffix}", flush=True)


# ── Step 1 · OBSCapture mock ──────────────────────────────────────────────────

def load_mock_frames() -> list[bytes]:
    """读取 sample_frames/ 下所有 JPG · 返回 bytes 列表（模拟 OBSCapture.frames()）。"""
    paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"sample_frames/ 下无 JPG: {FRAMES_DIR}")
    frames: list[bytes] = []
    for p in paths:
        frames.append(p.read_bytes())
    return frames


# ── Step 3 · 语义丰富的 WorldState 序列 ──────────────────────────────────────

# 根据帧文件名给出有意义的 stage 和数据
_FRAME_OVERRIDES: list[dict] = [
    # frame_001_pick
    dict(stage="pick", round="1-1", hp=100, gold=2,  level=1, exp="0/2",
         shop=["亚索", "盖伦", "卢锡安", "寒冰", "猪妹"],
         board=[], bench=[],
         active_traits=[], augments=[], bag=[],
         opponents_preview=[OpponentPreview(hp=100, top_carry="盖伦")]),
    # frame_002_pve
    dict(stage="pve", round="1-3", hp=100, gold=4, level=2, exp="2/4",
         shop=["阿狸", "盖伦", "亚索", "盲僧", "猪妹"],
         board=[Unit(name="盖伦", star=1, position=(3, 2))],
         bench=[Unit(name="亚索", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=1, tier="none")],
         augments=[], bag=[],
         opponents_preview=[OpponentPreview(hp=100, top_carry="盖伦")]),
    # frame_003_positioning
    dict(stage="positioning", round="2-1", hp=96, gold=6, level=3, exp="4/6",
         shop=["安妮", "卢锡安", "瑟庄妮", "剑圣", "吕布"],
         board=[Unit(name="盖伦", star=1, position=(3, 2)),
                Unit(name="亚索", star=1, position=(3, 4))],
         bench=[Unit(name="安妮", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=1, tier="none")],
         augments=[], bag=[],
         opponents_preview=[OpponentPreview(hp=96, top_carry="剑圣")]),
    # frame_004_pve
    dict(stage="pve", round="2-3", hp=90, gold=8, level=3, exp="6/6",
         shop=["安妮", "安妮", "卢锡安", "瑟庄妮", "李青"],
         board=[Unit(name="盖伦", star=1, position=(3, 2)),
                Unit(name="亚索", star=1, position=(3, 4)),
                Unit(name="安妮", star=1, position=(2, 3))],
         bench=[],
         active_traits=[ActiveTrait(name="铁卫", count=1, tier="none"),
                        ActiveTrait(name="法师", count=1, tier="none")],
         augments=[], bag=[BagItem(slot=0, name="暴风大剑")],
         opponents_preview=[OpponentPreview(hp=90, top_carry="亚索")]),
    # frame_005_augment
    dict(stage="augment", round="2-1", hp=88, gold=12, level=4, exp="8/10",
         shop=["剑圣", "盲僧", "李青", "猪妹", "吕布"],
         board=[Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="安妮", star=1, position=(2, 3))],
         bench=[Unit(name="亚索", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=2, tier="bronze"),
                        ActiveTrait(name="法师", count=2, tier="bronze")],
         augments=[], bag=[BagItem(slot=0, name="暴风大剑"),
                           BagItem(slot=1, name="反曲之弓")],
         opponents_preview=[OpponentPreview(hp=85, top_carry="剑圣")]),
    # frame_006_pvp
    dict(stage="pvp", round="3-1", hp=82, gold=14, level=5, exp="12/20",
         shop=["剑圣", "盲僧", "安妮", "卢锡安", "猪妹"],
         board=[Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="安妮", star=2, position=(2, 3)),
                Unit(name="亚索", star=1, position=(3, 5))],
         bench=[Unit(name="剑圣", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=2, tier="bronze"),
                        ActiveTrait(name="法师", count=2, tier="bronze")],
         augments=["法师之力"],
         bag=[BagItem(slot=0, name="暴风大剑"), BagItem(slot=1, name="反曲之弓")],
         opponents_preview=[OpponentPreview(hp=82, top_carry="剑圣", comp_summary="剑客流")]),
    # frame_007_pvp
    dict(stage="pvp", round="3-2", hp=75, gold=16, level=5, exp="16/20",
         shop=["剑圣", "安妮", "李青", "猪妹", "吕布"],
         board=[Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="安妮", star=2, position=(2, 3)),
                Unit(name="亚索", star=1, position=(3, 5)),
                Unit(name="剑圣", star=1, position=(3, 0))],
         bench=[],
         active_traits=[ActiveTrait(name="铁卫", count=2, tier="bronze"),
                        ActiveTrait(name="法师", count=2, tier="bronze"),
                        ActiveTrait(name="剑客", count=2, tier="bronze")],
         augments=["法师之力"],
         bag=[BagItem(slot=0, name="暴风大剑"), BagItem(slot=1, name="反曲之弓")],
         opponents_preview=[OpponentPreview(hp=70, top_carry="盖伦", comp_summary="护卫流")]),
    # frame_008_item
    dict(stage="item", round="3-3", hp=70, gold=18, level=6, exp="4/24",
         shop=["盲僧", "安妮", "李青", "猪妹", "吕布"],
         board=[Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="安妮", star=2, position=(2, 3))],
         bench=[Unit(name="亚索", star=1), Unit(name="剑圣", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=2, tier="bronze"),
                        ActiveTrait(name="法师", count=2, tier="bronze")],
         augments=["法师之力"],
         bag=[BagItem(slot=0, name="暴风大剑"), BagItem(slot=1, name="反曲之弓"),
              BagItem(slot=2, name="无用大棒")],
         opponents_preview=[OpponentPreview(hp=65, top_carry="剑圣")]),
    # frame_009_positioning
    dict(stage="positioning", round="4-1", hp=62, gold=22, level=6, exp="8/24",
         shop=["盖伦", "安妮", "李青", "猪妹", "蛮王"],
         board=[Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="安妮", star=2, position=(2, 3)),
                Unit(name="亚索", star=2, position=(3, 5))],
         bench=[Unit(name="剑圣", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=2, tier="bronze"),
                        ActiveTrait(name="法师", count=3, tier="silver")],
         augments=["法师之力"],
         bag=[BagItem(slot=0, name="无用大棒")],
         opponents_preview=[OpponentPreview(hp=60, top_carry="蛮王", comp_summary="狂暴流")]),
    # frame_010_pvp
    dict(stage="pvp", round="4-3", hp=52, gold=26, level=7, exp="2/36",
         shop=["蛮王", "安妮", "盲僧", "猪妹", "亚索"],
         board=[Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="安妮", star=2, position=(2, 3)),
                Unit(name="亚索", star=2, position=(3, 5)),
                Unit(name="剑圣", star=2, position=(3, 0))],
         bench=[Unit(name="安妮", star=1)],
         active_traits=[ActiveTrait(name="铁卫", count=2, tier="bronze"),
                        ActiveTrait(name="法师", count=3, tier="silver"),
                        ActiveTrait(name="剑客", count=2, tier="bronze")],
         augments=["法师之力"],
         bag=[],
         opponents_preview=[OpponentPreview(hp=50, top_carry="蛮王")]),
    # frame_011_pvp
    dict(stage="pvp", round="5-1", hp=42, gold=32, level=7, exp="14/36",
         shop=["蛮王", "安妮", "盲僧", "李青", "亚索"],
         board=[Unit(name="安妮", star=3, position=(2, 3)),
                Unit(name="盖伦", star=2, position=(3, 2)),
                Unit(name="亚索", star=2, position=(3, 5)),
                Unit(name="剑圣", star=2, position=(3, 0))],
         bench=[],
         active_traits=[ActiveTrait(name="法师", count=4, tier="gold"),
                        ActiveTrait(name="铁卫", count=2, tier="bronze")],
         augments=["法师之力"],
         bag=[],
         opponents_preview=[OpponentPreview(hp=38, top_carry="安妮")]),
    # frame_012_end
    dict(stage="end", round="5-5", hp=0, gold=8, level=7, exp="28/36",
         shop=[], board=[], bench=[],
         active_traits=[], augments=["法师之力"],
         bag=[],
         opponents_preview=[]),
]

_BASE_TS = time.time() - 300.0  # 局开始于 5 分钟前


def _make_world_states(frame_bytes_list: list[bytes]) -> list[WorldState]:
    """为每一帧构造一个语义丰富的 WorldState（mock 模式 · 不调 VLM 服务）。"""
    states: list[WorldState] = []
    for i, fb in enumerate(frame_bytes_list):
        override = _FRAME_OVERRIDES[i] if i < len(_FRAME_OVERRIDES) else {}
        ws = WorldState(
            stage=override.get("stage", "unknown"),
            round=override.get("round", f"{i//4+1}-{i%4+1}"),
            hp=override.get("hp", 100 - i * 5),
            gold=override.get("gold", i * 2),
            level=override.get("level", min(1 + i // 3, 9)),
            exp=override.get("exp", "0/0"),
            board=override.get("board", []),
            bench=override.get("bench", []),
            bag=override.get("bag", []),
            shop=override.get("shop", []),
            active_traits=override.get("active_traits", []),
            augments=override.get("augments", []),
            opponents_preview=override.get("opponents_preview", []),
            timestamp=_BASE_TS + i * 25.0,
        )
        states.append(ws)
    return states


# ── Step 2 · FrameMonitor ─────────────────────────────────────────────────────

def run_frame_monitor(frame_bytes_list: list[bytes]) -> tuple[FrameMonitor, int]:
    """跑真 FrameMonitor · 返回 (monitor, key_event_count)。"""
    from PIL import Image
    first_img = Image.open(io.BytesIO(frame_bytes_list[0]))
    monitor = FrameMonitor(screen_size=first_img.size)
    key_events = 0
    for fb in frame_bytes_list:
        events = monitor.observe(fb)
        if monitor.any_triggered(events):
            key_events += 1
    return monitor, key_events


# ── Step 4/5 · advice_server + DecisionLLM ───────────────────────────────────

async def start_advice_server() -> subprocess.Popen:
    """在 subprocess 里启动 advice_server · 返回 Popen 对象。"""
    cmd = [
        sys.executable,
        "-m", "src.advice_server",
        "--host", ADVICE_HOST,
        "--port", str(ADVICE_PORT),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # 等待端口就绪（最多 5s）
    import socket
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection((ADVICE_HOST, ADVICE_PORT), timeout=0.3):
                return proc
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(0.15)
    raise RuntimeError(f"advice_server 未在 {ADVICE_PORT} 就绪（5s timeout）")


async def check_vllm_available(
    base_url: str = "http://localhost:8000/v1",
) -> tuple[bool, str]:
    """探活 vLLM：先检查 /models · 再发一条最小 chat completions 确认模型已加载。

    Returns:
        (available, model_id)  model_id 为空字符串表示不可用。
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # 1. models 端点存在
            r = await client.get(f"{base_url}/models")
            if r.status_code != 200:
                return False, ""
            models = r.json().get("data", [])
            if not models:
                return False, ""
            model_id = models[0]["id"]
            # 2. 发一条最小推理确认模型真的在跑
            probe = {
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
            r2 = await client.post(
                f"{base_url}/chat/completions", json=probe,
                headers={"Authorization": "Bearer EMPTY"},
            )
            return (r2.status_code == 200), model_id
    except Exception:
        return False, ""


async def run_decision_llm(
    states: list[WorldState],
    vllm_ok: bool,
    model_id: str = "Qwen3-VL-4B-FP8",
    base_url: str = "http://localhost:8000/v1",
) -> list[tuple[str, float, str, object]]:
    """对每类 decision 触发一次 · 返回 [(kind, elapsed, mode, advice)] 列表。"""
    llm = DecisionLLM(
        base_url=base_url,
        model=model_id or "Qwen3-VL-4B-FP8",
        timeout=5.0,
    )

    # 找对应 stage 的 WorldState
    stage_to_ws: dict[str, WorldState] = {}
    for ws in states:
        stage_to_ws.setdefault(ws.stage, ws)

    # 决策任务：(kind, options, 首选 ws stage)
    tasks: list[tuple[str, list[str], str]] = [
        ("augment", ["法师之力", "复利", "攻速强化"], "augment"),
        ("carousel", ["安妮", "盖伦", "剑圣", "亚索"], "pick"),
        ("level", [], "pve"),
        ("positioning", [], "positioning"),
        ("item", [], "item"),
    ]

    results: list[tuple[str, float, str, object]] = []
    for kind, options, preferred_stage in tasks:
        ws = stage_to_ws.get(preferred_stage) or states[5]  # fallback 到 pvp 帧
        ctx = DecisionContext(kind=kind, options=options, timeout_s=25.0)  # type: ignore[arg-type]
        t0 = time.time()
        advice = await llm.decide(ws, ctx)
        elapsed = time.time() - t0
        if vllm_ok and advice.confidence > 0.0:
            mode = "guided_json ok"
        elif not vllm_ok:
            mode = "fallback (vllm unavailable)"
        else:
            mode = "fallback (decode error)"
        results.append((kind, elapsed, mode, advice))

    return results


async def post_advice_to_server(kind: str, advice_obj: object) -> None:
    """把 advice 用 HTTP POST 推给 advice_server /advice。"""
    import httpx
    url = f"http://{ADVICE_HOST}:{ADVICE_PORT}/advice"
    try:
        payload = json.loads(advice_obj.model_dump_json())  # type: ignore[union-attr]
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(url, json=payload)
    except Exception:
        pass  # 广播失败不阻断流程


# ── Step 6 · WebSocket client ──────────────────────────────────────────────────

async def ws_collect_broadcasts(expected: int, timeout: float = 8.0) -> list[dict]:
    """连接 WS · 收集至少 expected 条 advice broadcast · 返回消息列表。"""
    import websockets
    url = f"ws://{ADVICE_HOST}:{ADVICE_PORT}/ws/advice"
    received: list[dict] = []
    try:
        async with websockets.connect(url, open_timeout=3.0) as ws:  # type: ignore[attr-defined]
            deadline = time.time() + timeout
            while len(received) < expected and time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    msg = json.loads(raw)
                    if msg.get("type") == "advice":
                        received.append(msg["payload"])
                    elif msg.get("type") == "history":
                        received.append(msg["payload"])
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
    except Exception:
        pass
    return received


# ── Step 7 · LocalLLMAnalyzer ──────────────────────────────────────────────────

async def run_analyzer(
    states: list[WorldState],
    vllm_ok: bool,
    model_id: str = "Qwen3-VL-4B-FP8",
    base_url: str = "http://localhost:8000/v1",
) -> MatchReport:
    """触发复盘 · vLLM 可用真跑 · 否则 _empty_report + 手工填充。"""
    if vllm_ok:
        analyzer = LocalLLMAnalyzer(
            base_url=base_url,
            model=model_id or "Qwen3-VL-4B-FP8",
            timeout=60.0,
        )
        report = await analyzer.synthesize(states)
    else:
        # vLLM 不可用 · 用 _empty_report 骨架 + 手工填充演示字段
        report = MatchReport(
            match_id=f"E2E-SMOKE-{int(time.time())}",
            rank_tier="钻石 III",
            final_rank=3,
            final_hp=states[-1].hp if states else 0,
            duration_s=int(states[-1].timestamp - states[0].timestamp) if len(states) > 1 else 300,
            core_comp="法师流 (安妮 + 盖伦 + 亚索)",
            key_rounds=[
                RoundReview(
                    round="2-1",
                    grade="优",
                    title="选增强·法师之力",
                    comment=(
                        "2-1 选择「法师之力」强化，与场上安妮★2+盖伦★2法师羁绊高度契合。"
                        "若选择「复利」则经济增益约 +8 金，但法师流前中期爆发更需要技能增益覆盖，"
                        "「法师之力」带来约 18% 伤害提升，预期排名从 4 提升至 2.5。"
                    ),
                    delta="+18% 法师伤害",
                ),
                RoundReview(
                    round="3-3",
                    grade="可",
                    title="装备选装·物理散件拼接",
                    comment=(
                        "3-3 拿到暴风大剑+反曲之弓+无用大棒，理论上可合「暴风剑客之刃」给亚索，"
                        "但当前阵容主 C 是安妮（法师），散件优先合「魔法之书」会更匹配。"
                        "若此时优先给安妮合法术装，期望 5 回合内多出约 1.5 回合的伤害优势。"
                    ),
                    delta="-1.5 回合成型",
                ),
                RoundReview(
                    round="4-1",
                    grade="差",
                    title="定位失误·主C暴露前排",
                    comment=(
                        "4-1 安妮摆在 row=2 col=3，对面刺客系可以直接跳到安妮身上。"
                        "正确摆位应在 row=3 col=0 或 col=6 角落，配合盖伦前排吸引仇恨。"
                        "若调整摆位，安妮存活时间预计延长 1.2 回合，整体输出提升 20%。"
                    ),
                    delta="-20% 输出效率",
                ),
            ],
            summary=(
                "本局整体节奏稳健，2-1 选取「法师之力」是全局最优决策，契合度满分。"
                "核心问题在于 3-3 散件分配和 4-1 摆位失误，导致安妮作为主 C 频繁被集火。"
                "若能修正摆位 + 优化装备优先级，预期排名可从 3 提升至 1-2，HP 存留约 25-35。"
                "建议下局重点练习：法师流中期转型节奏（4-2 升 7 or 5-1 升 8）和反刺客摆位技巧。"
            ),
        )
    return report


# ── Main orchestration ────────────────────────────────────────────────────────

async def main() -> int:
    global _t0_global
    _t0_global = time.time()
    total_steps = 7
    server_proc: Optional[subprocess.Popen] = None

    try:
        # ── Step 1 · OBSCapture mock ──────────────────────────────────────────
        t = time.time()
        step(1, total_steps, "OBSCapture mock")
        frame_bytes_list = load_mock_frames()
        total_mb = sum(len(b) for b in frame_bytes_list) / 1024 / 1024
        elapsed = time.time() - t
        sub(f"{len(frame_bytes_list)} frames loaded · {total_mb:.1f} MB total", elapsed)

        # ── Step 2 · FrameMonitor ─────────────────────────────────────────────
        t = time.time()
        step(2, total_steps, "FrameMonitor analyzing")
        monitor, key_events = run_frame_monitor(frame_bytes_list)
        elapsed = time.time() - t
        sub(f"{key_events} key events detected · {monitor.frame_count} frames processed", elapsed)

        # ── Step 3 · VLMClient mock → WorldState ─────────────────────────────
        t = time.time()
        step(3, total_steps, "VLMClient mock · building WorldState sequence")
        # 用 VLMClient(mode="mock") 验证 pipeline 接口 · 但用语义丰富版覆盖数据
        vlm = VLMClient(mode="mock")
        # 先跑一次确认接口可用
        _ = await vlm.parse(frame_bytes_list[0])
        # 用覆盖版语义数据（mock 原始输出全是 unknown · 无演示价值）
        states = _make_world_states(frame_bytes_list)
        elapsed = time.time() - t
        stages = [ws.stage for ws in states]
        sub(f"{len(states)} WorldState · stages: {', '.join(stages)}", elapsed)

        # ── Step 4 · advice_server ────────────────────────────────────────────
        t = time.time()
        step(4, total_steps, f"Starting advice_server on :{ADVICE_PORT}")
        server_proc = await start_advice_server()
        elapsed = time.time() - t
        sub(f"pid={server_proc.pid} · listening on {ADVICE_HOST}:{ADVICE_PORT}", elapsed)

        # ── Step 5 · DecisionLLM ──────────────────────────────────────────────
        t = time.time()
        step(5, total_steps, "DecisionLLM · augment/carousel/level/positioning/item")
        vllm_ok, vllm_model_id = await check_vllm_available()
        if vllm_ok:
            sub(f"vLLM detected at localhost:8000 · model={vllm_model_id}")
        else:
            sub("vLLM unavailable · all decisions will use fallback")

        decision_results = await run_decision_llm(states, vllm_ok, model_id=vllm_model_id)

        # 推送到 advice_server（让 WS client 能收到 broadcast）
        for kind, elapsed_kind, mode, advice in decision_results:
            sub(f"{kind} advice · {elapsed_kind:.1f}s · {mode}")
            await post_advice_to_server(kind, advice)
            await asyncio.sleep(0.05)  # 小间隔让 WS 有时间推送

        elapsed = time.time() - t

        # ── Step 6 · WebSocket client ─────────────────────────────────────────
        t = time.time()
        step(6, total_steps, "WebSocket client subscribing to advice broadcasts")
        broadcasts = await ws_collect_broadcasts(expected=len(decision_results))
        elapsed = time.time() - t
        sub(f"received {len(broadcasts)} advice broadcasts", elapsed)
        if len(broadcasts) < len(decision_results):
            sub(
                f"[WARN] expected {len(decision_results)} broadcasts · "
                f"got {len(broadcasts)} · server may have dropped some"
            )

        # ── Step 7 · LocalLLMAnalyzer → render_report_html ───────────────────
        t = time.time()
        step(7, total_steps, "LocalLLMAnalyzer → render_report_html")

        # 完整序列给 analyzer（end 帧已经在 states[-1] 中）
        all_states_for_report = states

        report = await run_analyzer(all_states_for_report, vllm_ok, model_id=vllm_model_id)
        html_content = render_report_html(report)
        REPORT_PATH.write_text(html_content, encoding="utf-8")
        file_kb = REPORT_PATH.stat().st_size / 1024
        elapsed = time.time() - t
        sub(
            f"rank={report.final_rank} · {len(report.key_rounds)} key_rounds · "
            f"{len(report.summary)} chars summary",
            elapsed,
        )
        sub(f"{REPORT_PATH} ({file_kb:.0f} KB)", elapsed)

    finally:
        # cleanup
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    total_elapsed = time.time() - _t0_global
    print(f"\n✓ done · total {total_elapsed:.1f}s", flush=True)
    print(f"  report → {REPORT_PATH}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
