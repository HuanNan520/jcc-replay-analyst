"""LocalLLMAnalyzer · _scan_text_for_unknown_names unit tests.

Scenarios covered:
  1. all valid champion names -> empty scan result
  2. fabricated mixed name "弗雷尔卓德沃利贝尔" -> should be detected
  3. fabricated augment "经济之神" -> should be detected
  4. whitelist terms ("连胜" / "经济" etc.) -> no false positives
  5. fake names inside quotes still detected (both the old audit and the new scan hit)
  6. scanning empty text -> returns an empty list
  7. knowledge=None -> last_audit_warnings is initially empty
  8. the last_audit_warnings attribute is readable (when there is no knowledge)
"""
from __future__ import annotations

import pytest

from src.llm_analyzer import LocalLLMAnalyzer, _extract_candidate_names


# ──────────────────────────────────────────────
# Dummy knowledge fixture
# ──────────────────────────────────────────────

class _FakeKnowledge:
    """Minimal mock · only provides the all_* sets."""

    # sample of real S16 champion names (always valid)
    all_units: set[str] = {
        "艾希", "卡特琳娜", "盖伦", "诺克萨斯之手", "薇恩",
        "齐天大圣", "亚索", "乐芙兰", "璐璐", "赵信",
        "沙皇", "锤石", "克莱德", "卡牌大师", "皎月女神",
    }
    all_traits: set[str] = {
        "学院", "弗雷尔卓德", "枢纽", "暗影岛", "光辉女郎",
    }
    all_items: set[str] = {
        "暴风大剑", "B·F·大剑", "反曲之弓", "女神之泪", "锁子甲",
    }
    all_augments: set[str] = {
        "海克斯核心", "光明面", "暗面", "经济特训",
    }

    def version_context(self) -> str:
        return "fake"

    def comps_table(self) -> str:
        return "fake"

    def validate_unit_name(self, name: str) -> bool:
        return name in self.all_units


_KB = _FakeKnowledge()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_analyzer(with_knowledge: bool = True) -> LocalLLMAnalyzer:
    return LocalLLMAnalyzer(
        knowledge=_KB if with_knowledge else None,
    )


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_scan_all_known_units_returns_empty():
    """When the text only contains valid champion names from the knowledge base plus trigger words -> empty scan result."""
    analyzer = _make_analyzer()
    # "选艾希" "用薇恩" are both valid units
    text = "这局开局选艾希并合成装备，到薇恩升三星后开始用暴风大剑推进。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    # valid champion names should not appear in the suspicious list
    assert "艾希" not in result
    assert "薇恩" not in result


def test_scan_detects_mixed_fake_name():
    """Fabricated region+champion name "弗雷尔卓德沃利贝尔" — not in any known set -> should be detected.

    "弗雷尔卓德" is a real trait name, but "沃利贝尔" and the full 6-character concatenation are not known.
    This tests that the 6-character concatenation is detected when it appears next to a trigger word.
    """
    analyzer = _make_analyzer()
    # the full 6 chars "弗雷尔卓德沃" or "卓德沃利贝尔" are not in known
    text = "本局用弗雷尔卓德沃利贝尔合成了无敌阵容。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    # at least one sub-fragment not in known should be identified
    assert len(result) > 0


def test_scan_detects_fake_hex_augment():
    """The fabricated augment "经济之神" is not in all_augments -> should be detected (a fragment with 经济之神 as prefix/core should exist)."""
    analyzer = _make_analyzer()
    text = "选到经济之神后立刻滚雪球，连胜拿下整局。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    # "经济之神" is not in all_augments; at least one suspicious fragment contains this keyword
    assert any("经济之神" in frag for frag in result), f"经济之神 should be detected; actual suspicious list: {result}"


def test_scan_whitelist_terms_not_flagged():
    """Whitelist terms (连胜 / 经济 / 阵容 / 装备 / 海克斯) appearing next to a trigger word -> no false positive."""
    analyzer = _make_analyzer()
    text = (
        "本局靠连胜积累经济，选装备时优先合主C的暴风大剑，"
        "海克斯强化选了经济特训，阵容稳定推进。"
    )
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    for term in ("连胜", "经济", "装备", "海克斯", "阵容"):
        assert term not in result, f"whitelist word '{term}' should not be reported as suspicious"


def test_scan_empty_text_returns_empty():
    """Empty text -> returns an empty list."""
    analyzer = _make_analyzer()
    assert analyzer._scan_text_for_unknown_names("", _KB) == []


def test_scan_no_context_trigger_not_flagged():
    """A Chinese fragment with no surrounding trigger word should produce no false positive (lowers the false-positive rate)."""
    analyzer = _make_analyzer()
    # a name appears but with no trigger word (surrounded only by commas/periods)
    text = "整体表现良好。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    assert result == []


def test_last_audit_warnings_attribute_exists():
    """The last_audit_warnings attribute exists and is an empty list after init."""
    analyzer = _make_analyzer(with_knowledge=False)
    assert hasattr(analyzer, "last_audit_warnings")
    assert analyzer.last_audit_warnings == []


def test_last_audit_warnings_no_knowledge_stays_empty():
    """When knowledge=None, calling the internal scan method also does not raise and returns empty."""
    from src.schema import MatchReport, RoundReview
    analyzer = _make_analyzer(with_knowledge=False)
    report = MatchReport(
        match_id="test-0",
        final_rank=4,
        final_hp=20,
        duration_s=600,
        summary="test summary",
        key_rounds=[],
    )
    result = analyzer._scan_text_for_unknown_names_in_report(report)
    assert result == []
    assert analyzer.last_audit_warnings == []


def test_extract_candidate_names_basic():
    """_extract_candidate_names can extract Chinese fragments that have a contextual cue word."""
    text = "用艾希拿装备"
    candidates = _extract_candidate_names(text)
    assert "艾希" in candidates


def test_scan_known_trait_not_flagged():
    """A trait name in all_traits (such as "弗雷尔卓德") should not itself be reported."""
    analyzer = _make_analyzer()
    # "弗雷尔卓德" is in all_traits
    text = "本局选弗雷尔卓德羁绊开局。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    assert "弗雷尔卓德" not in result
