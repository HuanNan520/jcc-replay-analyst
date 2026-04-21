"""WorldState 数据结构 —— 一帧画面被识别后的结构化表达。

VLM / OCR / CV 层合流时都往这个 schema 里填 · 分析器拿它做复盘。
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

Stage = Literal[
    "pick",        # 选秀/选人
    "pve",         # 打小兵
    "pvp",         # 对战
    "augment",     # 选增强/海克斯
    "carousel",    # 轮抱
    "item",        # 选装备
    "positioning", # 摆位阶段
    "end",         # 局末结算
    "unknown",
]
TraitTier = Literal["bronze", "silver", "gold", "prismatic", "none"]


class Unit(BaseModel):
    name: str = Field(..., description="英雄中文名")
    star: int = Field(..., ge=1, le=3)
    items: List[str] = Field(default_factory=list)
    position: Optional[Tuple[int, int]] = Field(
        None, description="(行, 列) 棋盘坐标；备战区为 None"
    )


class ActiveTrait(BaseModel):
    name: str
    count: int = Field(..., ge=1)
    tier: TraitTier = "none"


class OpponentPreview(BaseModel):
    hp: int = Field(..., ge=0, le=100)
    top_carry: Optional[str] = None
    comp_summary: Optional[str] = None


class BagItem(BaseModel):
    """装备栏里的散装备组件。"""
    slot: int = Field(..., ge=0, le=9)
    name: str = Field(..., description="组件中文名：暴风大剑 / 反曲之弓 / 无用大棒 / ...")


class WorldState(BaseModel):
    """一帧画面识别后的结构化结果。"""
    stage: Stage
    round: str = Field(..., description="例: '3-2'")
    hp: int = Field(..., ge=0, le=100)
    gold: int = Field(..., ge=0)
    level: int = Field(..., ge=1, le=10)
    exp: str = Field(..., description="例: '12/20'")
    board: List[Unit] = Field(default_factory=list)
    bench: List[Unit] = Field(default_factory=list)
    bag: List[BagItem] = Field(default_factory=list)
    shop: List[str] = Field(default_factory=list)
    active_traits: List[ActiveTrait] = Field(default_factory=list)
    augments: List[str] = Field(default_factory=list)
    opponents_preview: List[OpponentPreview] = Field(default_factory=list)
    timestamp: float = Field(..., description="unix timestamp")


class RoundReview(BaseModel):
    """单回合复盘 —— 分析器输出的一行。"""
    round: str
    grade: Literal["优", "可", "差"]
    title: str = Field(..., description="这回合做的主要动作，例 '选增强 · 法师之力'")
    comment: str = Field(..., description="AI 给的点评 · 带因果分析")
    delta: Optional[str] = Field(
        None, description="量化影响，例 '+18% 伤害' / '-1.5 回合成型'"
    )


class MatchReport(BaseModel):
    """一局对局的完整复盘报告。"""
    match_id: str
    rank_tier: Optional[str] = None
    final_rank: int = Field(..., ge=1, le=8)
    final_hp: int = Field(..., ge=0)
    duration_s: int
    core_comp: Optional[str] = None
    key_rounds: List[RoundReview] = Field(default_factory=list)
    summary: str = Field(..., description="AI 对整局的一段总评 · 含改进建议")
