"""对局分析器 —— 把各感知层的输出串成一份完整复盘报告。

Pipeline:
  录屏 / 截图序列
    ↓
  frame_monitor   抽出关键帧（回合切换 / 商店刷新 / 战斗开始）
    ↓
  ┌────────────┬────────────┬────────────┐
  │  OCR 层    │   CV 层    │   VLM 层   │
  │ HP / 金币  │ 棋子 / 装备│ 羁绊 / 阵容│
  └────────────┴────────────┴────────────┘
    ↓
  合流成 WorldState 序列 · 每关键帧一个
    ↓
  LLM 分析器（带金铲铲版本知识 RAG） · 生成 MatchReport

本模块是骨架 · 每一层的 stub 都留了 TODO 标记 · 接上真实模型即可跑。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .frame_monitor import FrameMonitor
from .ocr_client import recognize, find_number_near
from .vlm_client import VLMClient
from .schema import MatchReport, RoundReview, WorldState

log = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_mode: str = "real"                # real / mock
    llm_mode: str = "mock"                # real / mock（未接入前先 mock）
    screen_w: int = 2560
    screen_h: int = 1456


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

    async def analyze_frames(self, frame_bytes_iter: Iterable[bytes]) -> MatchReport:
        """Main entry · 吃一个帧序列（bytes 迭代器）· 吐一份 MatchReport。"""
        key_states: List[WorldState] = []

        for i, frame in enumerate(frame_bytes_iter):
            # 1. 过滤 · 只留关键帧
            events = self.monitor.observe(frame)
            if not self.monitor.any_triggered(events) and i > 0:
                continue

            # 2. 三路识别
            ws = await self.vlm.parse(frame)            # VLM · 语义字段
            ws = self._overlay_ocr(frame, ws)           # OCR · 精确数字
            # TODO: CV 层 · 读装备图标 · 目前 VLM 已近似覆盖

            key_states.append(ws)
            log.info("frame %d · stage=%s · round=%s · hp=%d · gold=%d",
                     i, ws.stage, ws.round, ws.hp, ws.gold)

        # 3. LLM 分析
        report = await self._llm_synthesize(key_states)
        return report

    def _overlay_ocr(self, frame: bytes, ws: WorldState) -> WorldState:
        """用 OCR 覆盖 VLM 识别不准的数字字段（HP / gold / level）。"""
        try:
            hp = find_number_near(frame, "生命")
            if hp is not None and 0 <= hp <= 100:
                ws.hp = hp
        except Exception as e:
            log.debug("ocr hp miss: %s", e)

        try:
            gold = find_number_near(frame, "金币")
            if gold is not None and 0 <= gold <= 999:
                ws.gold = gold
        except Exception as e:
            log.debug("ocr gold miss: %s", e)

        return ws

    async def _llm_synthesize(self, states: List[WorldState]) -> MatchReport:
        """把状态序列交给 LLM · 生成评分和 narrative。

        TODO · 未接入前返回 placeholder。接入方式：
          - 每个 WorldState 序列化成简短 summary
          - 拼成一个大的 prompt + 当前金铲铲版本知识库（RAG）
          - 调 Claude API 或本地 vLLM 让它打分 + 写评语
          - 解析成 RoundReview 列表
        """
        if not states:
            return MatchReport(
                match_id="empty",
                final_rank=8, final_hp=0, duration_s=0,
                key_rounds=[],
                summary="空帧序列 · 无数据。",
            )

        # --- placeholder · 未接入 LLM 时的骨架输出 ---
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
                    comment="（TODO · 接入 LLM 后填充真实点评）",
                    delta=None,
                ) for ws in states[::max(len(states) // 5, 1)][:5]
            ],
            summary=(
                f"（骨架输出）识别到 {len(states)} 个关键帧 · "
                f"最终 HP {last.hp} · 等级 {last.level}。"
                " 接入 LLM 后此段将由 AI 生成带版本知识的叙事分析。"
            ),
        )
