"""实时 tick loop —— 闭环实时 coach 的中央枢纽。

数据流：
  OBSCapture.frames()  →  FrameMonitor.observe()  →  VLMClient.parse()
                                                         ↓
                                    _infer_decision_context(ws, prev_ws)
                                                         ↓
                                         DecisionLLM.decide(ws, ctx)
                                                         ↓
                                     AdvicePublisher.publish(advice)
                                                         ↓
                                       (ring_buffer 累积整局)
                                                         ↓
                                   [stage==end] → Analyzer.synthesize()
                                                         ↓
                                       reports/<match_id>-<ts>.md + .json

设计原则（严格遵守任务书）：
- 不自己写 LLM 调用 · 直接用 B3 的 DecisionLLM + 现有 LocalLLMAnalyzer
- 不引入 threading/queue · 全程 asyncio
- FrameMonitor 的关键帧触发是省算力核心 · 非触发帧不打 VLM
- 无 daemon / supervisor · 崩就崩 · 上层 shell/systemd 守
- ring_buffer 不落盘 · 完全内存 · 对局结束才落报告
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from .capture_obs import OBSCapture
from .frame_monitor import FrameMonitor, classify
from .vlm_client import VLMClient
from .schema import WorldState, MatchReport
from .decision_llm import DecisionLLM, DecisionContext, Advice
from .knowledge import load_knowledge
from .llm_analyzer import LocalLLMAnalyzer

log = logging.getLogger(__name__)


# ==================== Decision Context Inference ====================

# 升级到"下一级"所需金币门槛 —— 金铲铲累计经验表
# key = 目标等级 · value = 达到该等级累计需要的金币/经验
# 查询方式：升到 (level+1) 需要 LEVEL_THRESHOLDS[level+1] 金
# 注：任务书口径以"升到 level+1 要 N 金"衡量 · level=3 gold=10 → 升 4 级要 10 金
LEVEL_THRESHOLDS = {
    1: 0,
    2: 2,
    3: 6,
    4: 10,
    5: 20,
    6: 36,
    7: 56,
    8: 80,
    9: 96,
}


def _infer_decision_context(
    ws: WorldState,
    prev_ws: Optional[WorldState],
) -> Optional[DecisionContext]:
    """根据 WorldState 变化推断当前是否到了决策点 · 返回 DecisionContext 或 None。

    决策点判定规则（保守 · 宁可漏不可错判 · 六类）：

    1. augment: stage == 'augment' 且上一帧不是 augment（新弹出） ·
       options 从 ws.augments 最后三项或空占位 "?" 填充

    2. carousel: stage == 'carousel' 首次切入 · options 从 ws.shop 5 个棋子名取

    3. positioning: stage == 'positioning' 首次切入 · 单次触发 · 无 options

    4. level: stage 切换到 'pve' / 'pvp' 刚开始 · 且 gold >= 升级门槛 · 且 level < 9

    5. shop: 当前不触发 —— 商店每回合都在 · 触发太频繁 · 等用户需求确认再加
       (任务书注释：跳过 · 商店每回合都在 · 触发太频繁 · 等用户需求确认再加)

    6. item: bag 数量 >= 2 且比上帧多（出现新组件） · options 给所有 bag 组件名
    """
    # === 1. augment ===
    if ws.stage == "augment" and (prev_ws is None or prev_ws.stage != "augment"):
        opts = list(ws.augments[-3:]) if ws.augments else []
        # 补齐到 3 个 —— AugmentAdvice.ranked 期望 3 元素
        while len(opts) < 3:
            opts.append("?")
        return DecisionContext(kind="augment", options=opts)

    # === 2. carousel ===
    if ws.stage == "carousel" and (prev_ws is None or prev_ws.stage != "carousel"):
        return DecisionContext(kind="carousel", options=list(ws.shop[:5]))

    # === 3. positioning ===
    if ws.stage == "positioning" and (
        prev_ws is None or prev_ws.stage != "positioning"
    ):
        return DecisionContext(kind="positioning", options=[])

    # === 4. level ===
    # 进入战斗回合开始时 · 且经济够升到下一级门槛 · 且还没 9 级
    if ws.stage in ("pve", "pvp") and (
        prev_ws is None or prev_ws.stage not in ("pve", "pvp")
    ):
        required = LEVEL_THRESHOLDS.get(ws.level + 1, 999)
        if ws.gold >= required and ws.level < 9:
            return DecisionContext(kind="level", options=[])

    # === 5. shop · 显式不触发（见 docstring） ===

    # === 6. item ===
    # bag 变多（多出新组件）· 且总数 >= 2 · 才触发合成决策
    if (
        prev_ws is not None
        and len(ws.bag) >= 2
        and len(ws.bag) > len(prev_ws.bag)
    ):
        return DecisionContext(
            kind="item",
            options=[b.name for b in ws.bag],
        )

    return None


# ==================== Advice Publisher ====================

class AdvicePublisher:
    """POST /advice 给 B4 广播服务 · 失败不抛 · 仅 warn log。"""

    def __init__(self, server_url: str, timeout: float = 3.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AdvicePublisher":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *a) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def publish(self, advice: Advice) -> None:
        if self._client is None:
            # 允许不进 context manager 直接用 · 惰性起 client
            self._client = httpx.AsyncClient(timeout=self.timeout)
        try:
            r = await self._client.post(
                f"{self.server_url}/advice",
                json=advice.model_dump(),
            )
            r.raise_for_status()
            broadcast_to = -1
            try:
                broadcast_to = r.json().get("broadcast_to", -1)
            except Exception:
                pass
            log.debug(
                "advice published · kind=%s · broadcast_to=%s",
                advice.kind, broadcast_to,
            )
        except Exception as e:
            log.warning(
                "advice publish failed · kind=%s · %s",
                advice.kind, e,
            )


# ==================== Match Report Saver ====================

def _report_to_markdown(report: MatchReport) -> str:
    """把 MatchReport 渲染成 Markdown · 独立函数 · 方便测试。"""
    lines = [
        f"# 对局复盘 · {report.match_id}",
        "",
        f"- 最终排名：**{report.final_rank}** / 8",
        f"- 最终血量：{report.final_hp}",
        f"- 对局时长：{report.duration_s} 秒",
        f"- 核心阵容：{report.core_comp or '未识别'}",
    ]
    if report.rank_tier:
        lines.append(f"- 段位：{report.rank_tier}")
    lines += ["", "## 关键回合", ""]

    if not report.key_rounds:
        lines.append("_（无关键回合 · LLM 未识别出转折点）_")
        lines.append("")
    else:
        for r in report.key_rounds:
            grade_line = f"**评级：{r.grade}**"
            if r.delta:
                grade_line += f"　{r.delta}"
            lines += [
                f"### {r.round} · {r.title}",
                grade_line,
                "",
                r.comment,
                "",
            ]

    lines += ["## AI 总评", "", report.summary, ""]
    return "\n".join(lines)


def _save_report(report: MatchReport, reports_dir: Path) -> Path:
    """落盘 .md + .json 两份 · 返回 md 路径。"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md_path = reports_dir / f"{report.match_id}-{stamp}.md"
    json_path = md_path.with_suffix(".json")

    md_path.write_text(_report_to_markdown(report), encoding="utf-8")
    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    log.info("复盘已保存 · %s · %s", md_path, json_path)
    return md_path


# ==================== Live Tick Loop ====================

class LiveTickLoop:
    """实时 coach 中央枢纽 · asyncio 单循环。"""

    def __init__(
        self,
        capture: OBSCapture,
        vlm: VLMClient,
        decision_llm: DecisionLLM,
        post_match_llm: LocalLLMAnalyzer,
        publisher: AdvicePublisher,
        reports_dir: Path,
        ring_size: int = 240,   # 一局大概 30-40 分钟 · 2 fps 关键帧粗估 60-120 帧 · 留余量
        monitor: Optional[FrameMonitor] = None,
    ):
        self.capture = capture
        # screen_size 尽量用第一帧推 · 无法确认时默认 2560×1456（与 landscape 默认匹配）
        # 这里先 placeholder · 第一帧 observe 时会自己建 baseline
        self.monitor = monitor or FrameMonitor(screen_size=(2560, 1456))
        self.vlm = vlm
        self.decision_llm = decision_llm
        self.post_match_llm = post_match_llm
        self.publisher = publisher
        self.reports_dir = reports_dir
        self.ring: deque[WorldState] = deque(maxlen=ring_size)
        self._prev_ws: Optional[WorldState] = None
        self._match_started = False
        self._match_start_ts: Optional[float] = None

    async def run(self) -> None:
        log.info(
            "LiveTickLoop 启动 · reports_dir=%s · ring_size=%d",
            self.reports_dir, self.ring.maxlen,
        )
        async for frame in self.capture.frames():
            await self._process_frame(frame)

    async def _process_frame(self, frame: bytes) -> None:
        """单帧处理 · 拆出来方便测试 · 不直接调用 capture。"""
        t0 = time.time()
        events = self.monitor.observe(frame)

        # 对局进行中 · 无关键帧变化 → 跳过 · 省 VLM 调用
        if not self.monitor.any_triggered(events) and self._match_started:
            return

        event_kind = classify(self.monitor.changed_regions(events))
        log.debug("frame event=%s triggered=%d",
                  event_kind, sum(1 for e in events if e.triggered))

        # 感知 VLM
        try:
            ws = await self.vlm.parse(frame)
        except Exception as e:
            log.warning("VLM parse 失败 · %s", e)
            return

        # 无效状态过滤：unknown 且没开过局 → 忽略（还没进游戏）
        if ws.stage == "unknown" and not self._match_started:
            return

        # 首次见有效 stage · 标记对局开始
        if ws.stage != "unknown" and not self._match_started:
            self._match_started = True
            self._match_start_ts = ws.timestamp
            log.info(
                "对局开始 · round=%s stage=%s", ws.round, ws.stage,
            )

        self.ring.append(ws)

        # 对局结束 · 触发复盘（从非 end 切到 end）
        if (
            ws.stage == "end"
            and self._prev_ws is not None
            and self._prev_ws.stage != "end"
        ):
            await self._finalize_match()
            self._prev_ws = ws
            return

        # 决策点判断
        ctx = _infer_decision_context(ws, self._prev_ws)
        if ctx is not None:
            log.info(
                "决策点触发 · kind=%s round=%s stage=%s",
                ctx.kind, ws.round, ws.stage,
            )
            try:
                advice = await self.decision_llm.decide(ws, ctx)
                await self.publisher.publish(advice)
            except Exception as e:
                # DecisionLLM.decide 按约定永不抛 · publish 也吞异常 · 这里是兜底
                log.warning("决策链路异常 · %s", e)

        log.debug("frame processed · %.2fs", time.time() - t0)
        self._prev_ws = ws

    async def _finalize_match(self) -> None:
        """对局结束 · 拉 ring_buffer 调 post-match LLM · 落盘复盘。"""
        log.info(
            "对局结束 · 合成复盘 · ring_size=%d",
            len(self.ring),
        )
        try:
            report = await self.post_match_llm.synthesize(list(self.ring))
            _save_report(report, self.reports_dir)
        except Exception as e:
            log.error("复盘合成失败 · %s", e)

        # 重置 · 准备下一局
        self.ring.clear()
        self._match_started = False
        self._match_start_ts = None
        self._prev_ws = None


# ==================== CLI ====================

async def _main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m src.live_tick",
        description="jcc-coach 实时 tick loop · 闭环 coach + 对局结束自动复盘",
    )
    ap.add_argument(
        "--advice-server",
        default="http://localhost:8765",
        help="B4 advice_server 地址 · advice 会 POST 到这里广播给 overlay",
    )
    ap.add_argument(
        "--vlm-url",
        default="http://localhost:8000/v1",
        help="Qwen-VL vLLM 推理地址（感知层）",
    )
    ap.add_argument(
        "--vlm-model",
        default="Qwen3-VL-4B-FP8",
        help="Qwen-VL 模型名",
    )
    ap.add_argument(
        "--llm-url",
        default="http://localhost:8000/v1",
        help="Qwen vLLM 推理地址（决策 + 复盘层 · 可与 vlm-url 相同）",
    )
    ap.add_argument(
        "--llm-model",
        default="Qwen3-VL-4B-FP8",
        help="决策/复盘 LLM 模型名",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="OBS 帧抓取 fps",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="对局结束后 .md/.json 复盘存放目录",
    )
    ap.add_argument(
        "--season",
        default="s17",
        help="知识库赛季 · 默认 s17（国服当前）",
    )
    ap.add_argument(
        "--ring-size",
        type=int,
        default=240,
        help="ring_buffer 容量 · 关键帧上限",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别 · DEBUG/INFO/WARNING/ERROR",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    knowledge = load_knowledge(season=args.season)
    if knowledge is None:
        log.warning(
            "knowledge=%s 加载失败 · LLM 将降级到通用 TFT 规则",
            args.season,
        )

    capture = OBSCapture(fps=args.fps)
    vlm = VLMClient(
        base_url=args.vlm_url,
        model=args.vlm_model,
        mode="real",
    )
    decision_llm = DecisionLLM(
        base_url=args.llm_url,
        model=args.llm_model,
        knowledge=knowledge,
    )
    post_match = LocalLLMAnalyzer(
        base_url=args.llm_url,
        model=args.llm_model,
        knowledge=knowledge,
    )

    async with AdvicePublisher(args.advice_server) as publisher:
        loop = LiveTickLoop(
            capture=capture,
            vlm=vlm,
            decision_llm=decision_llm,
            post_match_llm=post_match,
            publisher=publisher,
            reports_dir=args.reports_dir,
            ring_size=args.ring_size,
        )
        await loop.run()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()


__all__ = [
    "LiveTickLoop",
    "AdvicePublisher",
    "_infer_decision_context",
    "_save_report",
    "_report_to_markdown",
    "LEVEL_THRESHOLDS",
]
