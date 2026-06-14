"""Unit tests · src/live_tick.py

Main coverage:
- the six decision points of `_infer_decision_context` plus negative cases (no trigger)
- the graceful behavior of `AdvicePublisher.publish` when B4 is available / unavailable (no exceptions)
- the output shape of `_report_to_markdown` + `_save_report`
- the core branches of `LiveTickLoop._process_frame` (with VLM / Capture / LLM / Publisher mocked out)

Not tested (require real services):
- OBSCapture itself · the real HTTP calls of DecisionLLM / LocalLLMAnalyzer
- full CLI start/stop
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from src.live_tick import (
    AdvicePublisher,
    LEVEL_THRESHOLDS,
    LiveTickLoop,
    _infer_decision_context,
    _report_to_markdown,
    _save_report,
)
from src.decision_llm import (
    AugmentAdvice,
    CarouselAdvice,
    DecisionContext,
    LevelAdvice,
    PositioningAdvice,
    ItemAdvice,
)
from src.schema import BagItem, MatchReport, RoundReview, WorldState


# ============================================================
# Helpers
# ============================================================

def _ws(
    stage: str = "pve",
    round: str = "1-1",
    hp: int = 100,
    gold: int = 0,
    level: int = 1,
    bag: Optional[list] = None,
    shop: Optional[list] = None,
    augments: Optional[list] = None,
) -> WorldState:
    return WorldState(
        stage=stage,  # type: ignore[arg-type]
        round=round,
        hp=hp,
        gold=gold,
        level=level,
        exp="0/0",
        timestamp=time.time(),
        bag=bag or [],
        shop=shop or [],
        augments=augments or [],
    )


def _bag(*names: str) -> list[BagItem]:
    return [BagItem(slot=i, name=n) for i, n in enumerate(names)]


def _png_bytes(color=(40, 40, 40), size=(64, 64)) -> bytes:
    """Synthesize a small PNG to feed FrameMonitor as input."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Six decision points —— _infer_decision_context
# ============================================================

class TestDecisionContextInference:
    # ---------- augment ----------
    def test_augment_first_entry(self):
        ctx = _infer_decision_context(_ws("augment"), prev_ws=None)
        assert ctx is not None
        assert ctx.kind == "augment"
        assert len(ctx.options) == 3   # always padded to 3

    def test_augment_with_three_augments_uses_last_three(self):
        ws = _ws(
            "augment",
            augments=["银色打野", "金色灼烧", "棱彩炼金"],
        )
        ctx = _infer_decision_context(ws, prev_ws=None)
        assert ctx is not None
        assert ctx.kind == "augment"
        assert ctx.options == ["银色打野", "金色灼烧", "棱彩炼金"]

    def test_augment_not_retriggered_if_still_augment(self):
        prev = _ws("augment")
        ctx = _infer_decision_context(_ws("augment"), prev_ws=prev)
        assert ctx is None

    def test_augment_after_leaving_then_returning_retriggers(self):
        # augment -> pve -> augment · should retrigger
        prev = _ws("pve")
        ctx = _infer_decision_context(_ws("augment"), prev_ws=prev)
        assert ctx is not None and ctx.kind == "augment"

    # ---------- carousel ----------
    def test_carousel_first_entry(self):
        ctx = _infer_decision_context(
            _ws("carousel", shop=["烬", "吉格斯", "寒冰", "蔚", "金克斯"]),
            prev_ws=None,
        )
        assert ctx is not None
        assert ctx.kind == "carousel"
        assert ctx.options == ["烬", "吉格斯", "寒冰", "蔚", "金克斯"]

    def test_carousel_not_retriggered(self):
        prev = _ws("carousel")
        ctx = _infer_decision_context(_ws("carousel"), prev_ws=prev)
        assert ctx is None

    # ---------- positioning ----------
    def test_positioning_first_entry(self):
        ctx = _infer_decision_context(
            _ws("positioning"),
            prev_ws=_ws("pvp"),
        )
        assert ctx is not None and ctx.kind == "positioning"
        assert ctx.options == []

    def test_positioning_not_retriggered(self):
        prev = _ws("positioning")
        ctx = _infer_decision_context(_ws("positioning"), prev_ws=prev)
        assert ctx is None

    # ---------- level ----------
    def test_level_triggered_when_gold_enough(self):
        prev = _ws("augment")
        # level=3 · the level-4 threshold = 10 gold
        curr = _ws("pve", gold=10, level=3)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is not None and ctx.kind == "level"

    def test_level_not_triggered_when_insufficient_gold(self):
        prev = _ws("augment")
        # level=3 gold=4 < 10 threshold · no trigger
        curr = _ws("pve", gold=4, level=3)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_level_capped_at_9(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=500, level=9)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_pve_to_pvp_not_retriggered(self):
        # already in a combat round · pve -> pvp should not retrigger level
        prev = _ws("pve", gold=10, level=3)
        curr = _ws("pvp", gold=8, level=4)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    # ---------- item ----------
    def test_item_triggered_on_new_bag_component(self):
        prev = _ws("pve", bag=_bag("暴风大剑"))
        curr = _ws("pve", bag=_bag("暴风大剑", "反曲之弓"))
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is not None and ctx.kind == "item"
        assert ctx.options == ["暴风大剑", "反曲之弓"]

    def test_item_not_triggered_if_bag_size_one(self):
        # only 1 component · not enough to combine · no trigger
        prev = _ws("pve", bag=[])
        curr = _ws("pve", bag=_bag("暴风大剑"))
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_item_not_triggered_if_bag_unchanged(self):
        prev = _ws("pve", bag=_bag("暴风大剑", "反曲之弓"))
        curr = _ws("pve", bag=_bag("暴风大剑", "反曲之弓"))
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    # ---------- idle ----------
    def test_no_trigger_on_pvp_steady(self):
        prev = _ws("pvp")
        curr = _ws("pvp")
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_no_trigger_on_unknown(self):
        ctx = _infer_decision_context(_ws("unknown"), prev_ws=None)
        assert ctx is None

    # ---------- shop explicitly skipped ----------
    def test_shop_kind_never_triggered_directly(self):
        # even if the shop has content when entering pick, do not take the shop branch
        prev = _ws("augment")
        curr = _ws("pick", shop=["卡特琳娜", "金克斯", "小炮", "风女", "亚索"])
        ctx = _infer_decision_context(curr, prev_ws=prev)
        # pick is not in augment/carousel/positioning/pve/pvp · should not trigger
        assert ctx is None


# ============================================================
# LEVEL_THRESHOLDS table self-check
# ============================================================

class TestLevelThresholds:
    def test_known_thresholds(self):
        assert LEVEL_THRESHOLDS[4] == 10
        assert LEVEL_THRESHOLDS[5] == 20
        assert LEVEL_THRESHOLDS[9] == 96

    def test_monotonic(self):
        prev = -1
        for lvl in sorted(LEVEL_THRESHOLDS):
            assert LEVEL_THRESHOLDS[lvl] > prev
            prev = LEVEL_THRESHOLDS[lvl]


# ============================================================
# _report_to_markdown / _save_report
# ============================================================

def _dummy_report(match_id: str = "TFT-S17-test") -> MatchReport:
    return MatchReport(
        match_id=match_id,
        final_rank=3,
        final_hp=22,
        duration_s=1840,
        core_comp="虚空 / 神谕",
        rank_tier="钻石 I",
        key_rounds=[
            RoundReview(
                round="3-2",
                grade="优",
                title="选增强 · 法师之力",
                comment="选择拉开主 C 伤害曲线 · 反事实：若错选利息强化 · 期望第 5。",
                delta="+18% 伤害",
            ),
        ],
        summary="整体节奏合理 · 4-1 连败略亏血 · 若提前留连败控血可望冲 2。",
    )


class TestReportRendering:
    def test_markdown_has_all_sections(self):
        md = _report_to_markdown(_dummy_report())
        assert "# 对局复盘 · TFT-S17-test" in md
        assert "最终排名：**3** / 8" in md
        assert "最终血量：22" in md
        assert "段位：钻石 I" in md
        assert "## 关键回合" in md
        assert "### 3-2 · 选增强 · 法师之力" in md
        assert "**评级：优**" in md
        assert "+18% 伤害" in md
        assert "## AI 总评" in md

    def test_markdown_handles_empty_rounds(self):
        report = _dummy_report()
        report.key_rounds = []
        md = _report_to_markdown(report)
        assert "无关键回合" in md
        assert "## AI 总评" in md

    def test_save_report_writes_md_and_json(self, tmp_path: Path):
        report = _dummy_report()
        md_path = _save_report(report, tmp_path)
        json_path = md_path.with_suffix(".json")

        assert md_path.exists()
        assert json_path.exists()
        assert md_path.read_text("utf-8").startswith("# 对局复盘")

        data = json.loads(json_path.read_text("utf-8"))
        assert data["match_id"] == "TFT-S17-test"
        assert data["final_rank"] == 3

    def test_save_report_creates_dir(self, tmp_path: Path):
        nested = tmp_path / "reports" / "subdir"
        assert not nested.exists()
        _save_report(_dummy_report(), nested)
        assert nested.is_dir()


# ============================================================
# AdvicePublisher —— graceful failure
# ============================================================

class TestAdvicePublisher:
    @pytest.mark.asyncio
    async def test_publish_swallows_network_error(self):
        """B4 offline · publish must not raise · must not crash the main loop."""
        pub = AdvicePublisher("http://127.0.0.1:1")   # port 1 · always refuses connection
        advice = LevelAdvice(
            reasoning="test",
            confidence=0.5,
            action="up",
        )
        async with pub:
            # should not raise
            await pub.publish(advice)

    @pytest.mark.asyncio
    async def test_publish_posts_correct_body(self, monkeypatch):
        """Success path · verify the body shape."""
        captured = {}

        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"broadcast_to": 2}

        async def fake_post(self, url, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        advice = AugmentAdvice(
            reasoning="ok",
            confidence=0.8,
            ranked=["A", "B", "C"],
            recommendation="A",
        )
        async with AdvicePublisher("http://example.com") as pub:
            await pub.publish(advice)

        assert captured["url"] == "http://example.com/advice"
        assert captured["json"]["kind"] == "augment"
        assert captured["json"]["recommendation"] == "A"

    @pytest.mark.asyncio
    async def test_publish_trailing_slash_normalized(self):
        pub = AdvicePublisher("http://example.com///")
        assert pub.server_url == "http://example.com"


# ============================================================
# LiveTickLoop._process_frame —— core state machine
# ============================================================

class _FakeVLM:
    """Return a preset WorldState by frame index · no real HTTP."""

    def __init__(self, states: list[WorldState]):
        self._states = list(states)
        self._idx = 0

    async def parse(self, _frame: bytes) -> WorldState:
        ws = self._states[self._idx]
        self._idx += 1
        return ws


class _RecordingPublisher:
    """Stand-in for AdvicePublisher · records every publish call."""

    def __init__(self):
        self.published: list = []

    async def publish(self, advice) -> None:
        self.published.append(advice)


class _RecordingDecisionLLM:
    """Stand-in for DecisionLLM · returns a placeholder Advice per kind · records every call."""

    def __init__(self):
        self.calls: list[DecisionContext] = []

    async def decide(self, ws: WorldState, ctx: DecisionContext):
        self.calls.append(ctx)
        if ctx.kind == "augment":
            return AugmentAdvice(
                reasoning="test", confidence=0.9,
                ranked=["x", "y", "z"], recommendation="x",
            )
        if ctx.kind == "carousel":
            return CarouselAdvice(
                reasoning="test", confidence=0.7,
                priority=["a"], recommendation="a",
            )
        if ctx.kind == "positioning":
            return PositioningAdvice(
                reasoning="test", confidence=0.5,
                main_carry_row=3, main_carry_col=3,
                bait_unit=None, notes=[],
            )
        if ctx.kind == "level":
            return LevelAdvice(
                reasoning="test", confidence=0.6,
                action="up", hold_gold_above=None,
            )
        if ctx.kind == "item":
            return ItemAdvice(
                reasoning="test", confidence=0.5,
                target_unit="?", combine=["暴风大剑", "反曲之弓"],
                hold_for_later=[],
            )
        raise ValueError(f"unknown kind: {ctx.kind}")


class _RecordingPostMatch:
    """Stand-in for LocalLLMAnalyzer · returns a dummy MatchReport · records states."""

    def __init__(self):
        self.states_seen: list[list[WorldState]] = []

    async def synthesize(self, states: list[WorldState]) -> MatchReport:
        self.states_seen.append(list(states))
        return _dummy_report(match_id="TFT-S17-live-test")


def _fresh_loop(
    states_for_vlm: list[WorldState],
    tmp_path: Path,
) -> tuple[LiveTickLoop, _RecordingDecisionLLM, _RecordingPublisher, _RecordingPostMatch]:
    """Build a LiveTickLoop with all dependencies mocked."""
    vlm = _FakeVLM(states_for_vlm)
    decision_llm = _RecordingDecisionLLM()
    post_match = _RecordingPostMatch()
    publisher = _RecordingPublisher()

    loop = LiveTickLoop(
        capture=None,        # type: ignore[arg-type]  does not call the real run()
        vlm=vlm,             # type: ignore[arg-type]
        decision_llm=decision_llm,  # type: ignore[arg-type]
        post_match_llm=post_match,  # type: ignore[arg-type]
        publisher=publisher,  # type: ignore[arg-type]
        reports_dir=tmp_path,
    )
    return loop, decision_llm, publisher, post_match


class TestProcessFrame:
    @pytest.mark.asyncio
    async def test_first_valid_frame_starts_match(self, tmp_path):
        """The first frame takes the VLM path (even with any_triggered=False) because _match_started=False means no short-circuit."""
        ws0 = _ws("pve", round="1-1", gold=0, level=1)
        loop, _dec, _pub, _ = _fresh_loop([ws0], tmp_path)

        frame = _png_bytes()
        await loop._process_frame(frame)
        # VLM returns a valid stage · should mark the match as started · ring has one entry
        assert loop._match_started is True
        assert len(loop.ring) == 1
        assert loop.ring[-1].stage == "pve"

    @pytest.mark.asyncio
    async def test_trigger_augment_decision_on_first_frame(self, tmp_path):
        """First frame VLM yields augment · because prev_ws=None -> triggers the augment decision."""
        ws_augment = _ws("augment", round="2-1", augments=["A", "B", "C"])
        loop, decision_llm, publisher, _ = _fresh_loop(
            [ws_augment], tmp_path,
        )

        frame = _png_bytes(color=(240, 180, 120))
        await loop._process_frame(frame)

        # VLM called · WS added to ring · augment decision triggered
        assert loop._match_started is True
        assert len(loop.ring) == 1
        assert loop.ring[-1].stage == "augment"
        assert len(decision_llm.calls) == 1
        assert decision_llm.calls[0].kind == "augment"
        assert len(publisher.published) == 1
        assert publisher.published[0].kind == "augment"

    @pytest.mark.asyncio
    async def test_end_stage_finalizes_match(self, tmp_path):
        """stage goes pvp -> end · triggers replay synthesis · generates md/json · clears the ring."""
        ws_pvp = _ws("pvp", round="5-3")
        ws_end = _ws("end", round="5-4", hp=0)
        loop, _dec, _pub, post_match = _fresh_loop(
            [ws_pvp, ws_end], tmp_path,
        )

        # first frame · VLM yields pvp · match starts
        f0 = _png_bytes(color=(10, 10, 10))
        await loop._process_frame(f0)
        assert loop._match_started is True
        assert len(loop.ring) == 1

        # second frame · ROI changes · VLM yields end · triggers finalize
        f1 = _png_bytes(color=(240, 180, 120))
        await loop._process_frame(f1)

        # after finalize · ring cleared · match_started reset
        assert loop._match_started is False
        assert len(loop.ring) == 0
        # post-match LLM called · and received both pvp+end states
        assert len(post_match.states_seen) == 1
        assert len(post_match.states_seen[0]) == 2

        # md/json actually produced under the reports directory
        md_files = list(tmp_path.glob("*.md"))
        json_files = list(tmp_path.glob("*.json"))
        assert len(md_files) == 1
        assert len(json_files) == 1

    @pytest.mark.asyncio
    async def test_unknown_before_match_start_is_ignored(self, tmp_path):
        """VLM returns unknown and no match has started yet · does not enter the ring or call the decision."""
        ws_unknown_a = _ws("unknown")
        ws_unknown_b = _ws("unknown")
        loop, decision_llm, publisher, _ = _fresh_loop(
            [ws_unknown_a, ws_unknown_b], tmp_path,
        )

        f0 = _png_bytes(color=(10, 10, 10))
        f1 = _png_bytes(color=(240, 180, 120))
        await loop._process_frame(f0)
        await loop._process_frame(f1)

        assert loop._match_started is False
        assert len(loop.ring) == 0
        assert len(decision_llm.calls) == 0
        assert len(publisher.published) == 0

    @pytest.mark.asyncio
    async def test_no_trigger_short_circuits_when_match_started(self, tmp_path):
        """Match in progress · ROI unchanged -> skip directly · no VLM call · ring does not grow."""
        ws_pvp = _ws("pvp", round="3-2")
        loop, _dec, _pub, _ = _fresh_loop(
            [ws_pvp], tmp_path,
        )
        # manually mark the loop as mid-match
        loop._match_started = True

        same_frame = _png_bytes(color=(10, 10, 10))
        # first frame baseline (all triggered=False) · but match_started=True · takes the short-circuit return
        await loop._process_frame(same_frame)
        # second frame, same color · keeps short-circuiting
        await loop._process_frame(same_frame)
        # VLM should never have been called · ring did not grow
        assert len(loop.ring) == 0
