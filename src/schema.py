"""WorldState data structures — the structured representation of one recognized frame.

The VLM / OCR / CV layers all write into this schema when they merge · the analyzer
consumes it for replay analysis.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

Stage = Literal[
    "pick",        # draft / champion pick
    "pve",         # fighting minions (PvE rounds)
    "pvp",         # player-vs-player combat
    "augment",     # choosing an augment
    "carousel",    # shared carousel
    "item",        # choosing an item
    "positioning", # positioning phase
    "end",         # end-of-game settlement
    "unknown",
]
TraitTier = Literal["bronze", "silver", "gold", "prismatic", "none"]


class Unit(BaseModel):
    name: str = Field(..., description="champion name (Chinese, as shown in-game)")
    star: int = Field(..., ge=1, le=3)
    items: List[str] = Field(default_factory=list)
    position: Optional[Tuple[int, int]] = Field(
        None, description="(row, col) board coordinate; None when on the bench"
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
    """A loose item component sitting in the item bag."""
    slot: int = Field(..., ge=0, le=9)
    name: str = Field(..., description="component name (Chinese in-game): 暴风大剑 / 反曲之弓 / 无用大棒 / ...")


class WorldState(BaseModel):
    """The structured result of recognizing one frame."""
    stage: Stage
    round: str = Field(..., description="e.g. '3-2'")
    hp: int = Field(..., ge=0, le=100)
    gold: int = Field(..., ge=0)
    level: int = Field(..., ge=1, le=10)
    exp: str = Field(..., description="e.g. '12/20'")
    board: List[Unit] = Field(default_factory=list)
    bench: List[Unit] = Field(default_factory=list)
    bag: List[BagItem] = Field(default_factory=list)
    shop: List[str] = Field(default_factory=list)
    active_traits: List[ActiveTrait] = Field(default_factory=list)
    augments: List[str] = Field(default_factory=list)
    opponents_preview: List[OpponentPreview] = Field(default_factory=list)
    timestamp: float = Field(..., description="unix timestamp")


class RoundReview(BaseModel):
    """A single-round review — one row of the analyzer's output.

    NOTE: the grade values and the Field descriptions below are emitted into
    MatchReport.model_json_schema() and fed to the LLM as guided_json · they stay
    Chinese so the (Chinese-coaching) model's output schema stays coherent.
    """
    round: str
    # grade values matched against LLM output and keyed in _GRADE_CSS / coercion logic
    # (优 = good, 可 = ok, 差 = poor).
    grade: Literal["优", "可", "差"]
    title: str = Field(..., description="这回合做的主要动作，例 '选增强 · 法师之力'")
    comment: str = Field(..., description="AI 给的点评 · 带因果分析")
    delta: Optional[str] = Field(
        None, description="量化影响，例 '+18% 伤害' / '-1.5 回合成型'"
    )


class MatchReport(BaseModel):
    """The complete replay-analysis report for one match.

    NOTE: this schema is fed to the LLM as guided_json · the summary description below
    stays Chinese so the model's output schema stays coherent.
    """
    match_id: str
    rank_tier: Optional[str] = None
    final_rank: int = Field(..., ge=1, le=8)
    final_hp: int = Field(..., ge=0)
    duration_s: int
    core_comp: Optional[str] = None
    key_rounds: List[RoundReview] = Field(default_factory=list)
    summary: str = Field(..., description="AI 对整局的一段总评 · 含改进建议")
