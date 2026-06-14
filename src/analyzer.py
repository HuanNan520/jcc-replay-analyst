"""Match analyzer — stitches the output of the perception layers into a complete replay report.

Pipeline:
  recording / screenshot sequence
    v
  frame_monitor   extract keyframes (round transition / shop refresh / combat start)
    v
  +------------+------------+------------+
  | OCR layer  | CV layer   | VLM layer  |
  | HP / gold  | unit / item| trait/comp |
  +------------+------------+------------+
    v
  merged into a WorldState sequence · one per keyframe
    v
  LLM analyzer (with TFT version-knowledge RAG) · produces a MatchReport

This module is the pipeline orchestration of the replay path · both perception and LLM
are wired to real implementations:
  - VLMClient (vlm_client.py) runs on local vLLM Qwen3-VL
  - LocalLLMAnalyzer (llm_analyzer.py) runs on local vLLM guided_json
  - S17 KnowledgeProvider (knowledge.py) pulls version data from jcc-daida

For the real-time coach path see src/live_tick.py · it shares the same perception + LLM layers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .frame_monitor import FrameMonitor
from .knowledge import S16Knowledge, load_s16_knowledge
from .ocr_client import recognize, find_number_near
from .vlm_client import VLMClient
from .schema import MatchReport, RoundReview, WorldState

log = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_mode: str = "real"                # real / mock
    llm_mode: str = "mock"                # real / mock (mock until the LLM is wired up)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen3-VL-8B-FP8"
    screen_w: int = 2560
    screen_h: int = 1456
    enable_knowledge: bool = True         # try to load the S16 knowledge base · silently degrade on failure


class Analyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.cfg = config or AnalyzerConfig()
        self.vlm = VLMClient(
            base_url=self.cfg.vlm_base_url,
            model=self.cfg.vlm_model,
            mode=self.cfg.vlm_mode,
        )
        self.monitor = FrameMonitor(
            screen_size=(self.cfg.screen_w, self.cfg.screen_h),
        )
        self.knowledge: Optional[S16Knowledge] = (
            load_s16_knowledge() if self.cfg.enable_knowledge else None
        )
        if self.knowledge is not None:
            log.info(
                "S16 knowledge base loaded · %d comps / %d champions / %d traits",
                len(self.knowledge.comps),
                len(self.knowledge.all_units),
                len(self.knowledge.all_traits),
            )

    async def analyze_frames(self, frame_bytes_iter: Iterable[bytes]) -> MatchReport:
        """Main entry · takes a frame sequence (bytes iterator) · returns one MatchReport."""
        key_states: List[WorldState] = []

        for i, frame in enumerate(frame_bytes_iter):
            # 1. Filter · keep keyframes only
            events = self.monitor.observe(frame)
            if not self.monitor.any_triggered(events) and i > 0:
                continue

            # 2. Three-way recognition
            ws = await self.vlm.parse(frame)            # VLM · semantic fields
            ws = self._overlay_ocr(frame, ws)           # OCR · precise numbers
            # TODO: CV layer · read item icons · for now the VLM covers this approximately

            key_states.append(ws)
            log.info("frame %d · stage=%s · round=%s · hp=%d · gold=%d",
                     i, ws.stage, ws.round, ws.hp, ws.gold)

        # 3. LLM analysis
        report = await self._llm_synthesize(key_states)
        return report

    def _overlay_ocr(self, frame: bytes, ws: WorldState) -> WorldState:
        """Use OCR to override the numeric fields the VLM reads imprecisely (HP / gold / level)."""
        try:
            # "生命" (HP) is the on-screen Chinese label OCR anchors on — must stay Chinese.
            hp = find_number_near(frame, "生命")
            if hp is not None and 0 <= hp <= 100:
                ws.hp = hp
        except Exception as e:
            log.debug("ocr hp miss: %s", e)

        try:
            # "金币" (gold) is the on-screen Chinese label OCR anchors on — must stay Chinese.
            gold = find_number_near(frame, "金币")
            if gold is not None and 0 <= gold <= 999:
                ws.gold = gold
        except Exception as e:
            log.debug("ocr gold miss: %s", e)

        return ws

    async def _llm_synthesize(self, states: List[WorldState]) -> MatchReport:
        """Hand the state sequence to the LLM · produce grades and narrative.

        With llm_mode="real" this runs on local vLLM (LocalLLMAnalyzer) · with "mock" it
        returns a skeleton. self.knowledge implements the KnowledgeProvider Protocol
        (version_context / comps_table / validate_unit_name) · pass it via duck typing.
        When None, LocalLLMAnalyzer degrades to generic TFT rules on its own.
        """
        if not states:
            # Chinese report content kept verbatim — this is product-facing report text.
            return MatchReport(
                match_id="empty",
                final_rank=8, final_hp=0, duration_s=0,
                key_rounds=[],
                summary="空帧序列 · 无数据。",
            )

        if self.cfg.llm_mode == "mock":
            return self._mock_placeholder(states)

        from .llm_analyzer import LocalLLMAnalyzer
        llm = LocalLLMAnalyzer(
            base_url=self.cfg.llm_base_url,
            model=self.cfg.llm_model,
            knowledge=self.knowledge,
        )
        return await llm.synthesize(states)

    def _mock_placeholder(self, states: List[WorldState]) -> MatchReport:
        """Skeleton output for mock mode · never reached when llm_mode='real'.

        The grade/title/comment/summary strings below are Chinese report content the
        product emits — kept verbatim so the mock report reads like the real one.
        """
        first, last = states[0], states[-1]
        match_id = f"TFT-{int(time.time())}"
        return MatchReport(
            match_id=match_id,
            rank_tier=None,
            final_rank=4,
            final_hp=last.hp,
            duration_s=int(last.timestamp - first.timestamp) if len(states) > 1 else 0,
            core_comp=(",".join(t.name for t in last.active_traits[:2]) or None),
            key_rounds=[
                RoundReview(
                    round=ws.round, grade="可",
                    title=f"{ws.stage} · 级 {ws.level} · 金 {ws.gold}",
                    comment="（mock 模式 · 未调 LLM · 接 --llm real 获取真实点评）",
                    delta=None,
                ) for ws in states[::max(len(states) // 5, 1)][:5]
            ],
            summary=(
                f"（mock 骨架）识别到 {len(states)} 个关键帧 · "
                f"最终 HP {last.hp} · 等级 {last.level}。"
                " 切 --llm real 走本地 vLLM 生成带版本知识的叙事分析。"
            ),
        )
