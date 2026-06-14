"""Real-time tick loop — the central hub of the closed-loop real-time coach.

Data flow:
  OBSCapture.frames()  ->  FrameMonitor.observe()  ->  VLMClient.parse()
                                                         v
                                    _infer_decision_context(ws, prev_ws)
                                                         v
                                         DecisionLLM.decide(ws, ctx)
                                                         v
                                     AdvicePublisher.publish(advice)
                                                         v
                                       (ring_buffer accumulates the whole match)
                                                         v
                                   [stage==end] -> Analyzer.synthesize()
                                                         v
                                       reports/<match_id>-<ts>.md + .json

Design principles:
- Do not write LLM calls here · use DecisionLLM + the existing LocalLLMAnalyzer directly
- No threading/queue · asyncio throughout
- FrameMonitor keyframe triggering is the core compute saver · non-triggered frames never hit the VLM
- No daemon / supervisor · let it crash · an outer shell/systemd keeps watch
- ring_buffer is never persisted · fully in-memory · only the report is written when the match ends
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
from .report_renderer import render_report_html

log = logging.getLogger(__name__)


# ==================== Decision Context Inference ====================

# Gold threshold required to level up to the "next level" — TFT cumulative-experience table
# key = target level · value = the cumulative gold/experience needed to reach that level
# Lookup: leveling to (level+1) requires LEVEL_THRESHOLDS[level+1] gold
# Note: measured as "N gold to level up to level+1" · level=3 gold=10 -> 10 gold to reach level 4
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
    """Infer from WorldState changes whether we have reached a decision point · returns a DecisionContext or None.

    Decision-point rules (conservative · prefer missing over misfiring · six kinds):

    1. augment: stage == 'augment' and the previous frame was not augment (newly popped up) ·
       options filled from the last three of ws.augments, or "?" placeholders

    2. carousel: first entry into stage == 'carousel' · options taken from the 5 unit names in ws.shop

    3. positioning: first entry into stage == 'positioning' · fires once · no options

    4. level: stage just switched to 'pve' / 'pvp' · and gold >= the level-up threshold · and level < 9

    5. shop: currently never fires — the shop is present every round · would fire too often · revisit when needed

    6. item: bag count >= 2 and larger than the previous frame (a new component appeared) · options are all bag component names
    """
    # === 1. augment ===
    if ws.stage == "augment" and (prev_ws is None or prev_ws.stage != "augment"):
        opts = list(ws.augments[-3:]) if ws.augments else []
        # Pad to 3 — AugmentAdvice.ranked expects 3 elements
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
    # At the start of a combat round · with enough economy to hit the next level-up threshold · and not yet level 9
    if ws.stage in ("pve", "pvp") and (
        prev_ws is None or prev_ws.stage not in ("pve", "pvp")
    ):
        required = LEVEL_THRESHOLDS.get(ws.level + 1, 999)
        if ws.gold >= required and ws.level < 9:
            return DecisionContext(kind="level", options=[])

    # === 5. shop · explicitly never fires (see docstring) ===

    # === 6. item ===
    # The bag grew (a new component appeared) · and the total is >= 2 · only then trigger a combine decision
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
    """POST /advice to the broadcast service · never raises on failure · only logs a warning."""

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
            # Allow direct use without entering the context manager · lazily start the client
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
    """Render a MatchReport to Markdown · standalone function · easy to test.

    The Chinese strings below are the replay-report template the product emits · kept Chinese.
    """
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
    """Write three files (.md + .json + .html) to disk · returns the md path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md_path = reports_dir / f"{report.match_id}-{stamp}.md"
    json_path = md_path.with_suffix(".json")
    html_path = md_path.with_suffix(".html")

    md_path.write_text(_report_to_markdown(report), encoding="utf-8")
    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    html_path.write_text(render_report_html(report), encoding="utf-8")
    log.info("replay saved · %s · %s · %s", md_path, json_path, html_path)
    return md_path


# ==================== Live Tick Loop ====================

class LiveTickLoop:
    """Central hub of the real-time coach · a single asyncio loop."""

    def __init__(
        self,
        capture: OBSCapture,
        vlm: VLMClient,
        decision_llm: DecisionLLM,
        post_match_llm: LocalLLMAnalyzer,
        publisher: AdvicePublisher,
        reports_dir: Path,
        ring_size: int = 240,   # a match is roughly 30-40 min · at 2 fps, ~60-120 keyframes · with headroom
        monitor: Optional[FrameMonitor] = None,
    ):
        self.capture = capture
        # Infer screen_size from the first frame when possible · default 2560x1456 when unknown (matches the landscape default)
        # This is a placeholder · the first observe() establishes the baseline itself
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
            "LiveTickLoop started · reports_dir=%s · ring_size=%d",
            self.reports_dir, self.ring.maxlen,
        )
        async for frame in self.capture.frames():
            await self._process_frame(frame)

    async def _process_frame(self, frame: bytes) -> None:
        """Process a single frame · factored out for testability · does not call capture directly."""
        t0 = time.time()
        events = self.monitor.observe(frame)

        # Mid-match · no keyframe change -> skip · saves a VLM call
        if not self.monitor.any_triggered(events) and self._match_started:
            return

        event_kind = classify(self.monitor.changed_regions(events))
        log.debug("frame event=%s triggered=%d",
                  event_kind, sum(1 for e in events if e.triggered))

        # Perception VLM
        try:
            ws = await self.vlm.parse(frame)
        except Exception as e:
            log.warning("VLM parse failed · %s", e)
            return

        # Invalid-state filter: unknown and no match started yet -> ignore (not in a game yet)
        if ws.stage == "unknown" and not self._match_started:
            return

        # First valid stage seen · mark the match as started
        if ws.stage != "unknown" and not self._match_started:
            self._match_started = True
            self._match_start_ts = ws.timestamp
            log.info(
                "match started · round=%s stage=%s", ws.round, ws.stage,
            )

        self.ring.append(ws)

        # Match ended · trigger replay analysis (transition from non-end to end)
        if (
            ws.stage == "end"
            and self._prev_ws is not None
            and self._prev_ws.stage != "end"
        ):
            await self._finalize_match()
            self._prev_ws = ws
            return

        # Decision-point check
        ctx = _infer_decision_context(ws, self._prev_ws)
        if ctx is not None:
            log.info(
                "decision point triggered · kind=%s round=%s stage=%s",
                ctx.kind, ws.round, ws.stage,
            )
            try:
                advice = await self.decision_llm.decide(ws, ctx)
                await self.publisher.publish(advice)
            except Exception as e:
                # DecisionLLM.decide never raises by contract · publish also swallows exceptions · this is a last-resort guard
                log.warning("decision pipeline error · %s", e)

        log.debug("frame processed · %.2fs", time.time() - t0)
        self._prev_ws = ws

    async def _finalize_match(self) -> None:
        """Match ended · pull the ring_buffer, call the post-match LLM · write the replay to disk."""
        log.info(
            "match ended · synthesizing replay · ring_size=%d",
            len(self.ring),
        )
        try:
            report = await self.post_match_llm.synthesize(list(self.ring))
            _save_report(report, self.reports_dir)
        except Exception as e:
            log.error("replay synthesis failed · %s", e)

        # Reset · ready for the next match
        self.ring.clear()
        self._match_started = False
        self._match_start_ts = None
        self._prev_ws = None


# ==================== CLI ====================

async def _main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m src.live_tick",
        description="jcc-coach real-time tick loop · closed-loop coach + automatic post-match replay analysis",
    )
    ap.add_argument(
        "--advice-server",
        default="http://localhost:8765",
        help="advice_server address · advice is POSTed here to be broadcast to the overlay",
    )
    ap.add_argument(
        "--vlm-url",
        default="http://localhost:8000/v1",
        help="Qwen-VL vLLM inference address (perception layer)",
    )
    ap.add_argument(
        "--vlm-model",
        default="Qwen3-VL-4B-FP8",
        help="Qwen-VL model name",
    )
    ap.add_argument(
        "--llm-url",
        default="http://localhost:8000/v1",
        help="Qwen vLLM inference address (decision + replay layer · may equal vlm-url)",
    )
    ap.add_argument(
        "--llm-model",
        default="Qwen3-VL-4B-FP8",
        help="decision/replay LLM model name",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="OBS frame-grab fps",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="directory where post-match .md/.json replays are stored",
    )
    ap.add_argument(
        "--season",
        default="s17",
        help="knowledge-base season · default s17 (current China server)",
    )
    ap.add_argument(
        "--ring-size",
        type=int,
        default=240,
        help="ring_buffer capacity · keyframe cap",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="log level · DEBUG/INFO/WARNING/ERROR",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    knowledge = load_knowledge(season=args.season)
    if knowledge is None:
        log.warning(
            "knowledge=%s failed to load · the LLM will degrade to generic TFT rules",
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
