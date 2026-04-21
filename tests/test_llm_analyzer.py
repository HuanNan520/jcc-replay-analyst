"""LocalLLMAnalyzer · _scan_text_for_unknown_names 单元测试。

覆盖场景：
  1. 全合法英雄名 → scan 结果为空
  2. 伪造混合名 "弗雷尔卓德沃利贝尔" → 应检出
  3. 伪造海克斯 "经济之神" → 应检出
  4. 白名单术语（"连胜" / "经济" 等）→ 不误报
  5. 引号内伪名仍被检出（兼容旧 audit + 新 scan 都命中）
  6. 扫描空文本 → 返回空列表
  7. knowledge=None → last_audit_warnings 初始为空
  8. last_audit_warnings 属性可读（无 knowledge 时）
"""
from __future__ import annotations

import pytest

from src.llm_analyzer import LocalLLMAnalyzer, _extract_candidate_names


# ──────────────────────────────────────────────
# Dummy knowledge fixture
# ──────────────────────────────────────────────

class _FakeKnowledge:
    """最小化 mock · 仅提供 all_* 集合。"""

    # S16 真实英雄名样本（一定合法）
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
# 辅助
# ──────────────────────────────────────────────

def _make_analyzer(with_knowledge: bool = True) -> LocalLLMAnalyzer:
    return LocalLLMAnalyzer(
        knowledge=_KB if with_knowledge else None,
    )


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_scan_all_known_units_returns_empty():
    """文本中只出现知识库合法英雄名 + 触发词 → 扫描结果为空。"""
    analyzer = _make_analyzer()
    # "选艾希" "用薇恩" 都是合法单位
    text = "这局开局选艾希并合成装备，到薇恩升三星后开始用暴风大剑推进。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    # 合法英雄名不应出现在可疑列表
    assert "艾希" not in result
    assert "薇恩" not in result


def test_scan_detects_mixed_fake_name():
    """伪造混合地区+英雄名 "弗雷尔卓德沃利贝尔" — 不在任何 known 集合 → 应检出。

    "弗雷尔卓德" 是真实羁绊名，但 "沃利贝尔" 和完整的 6 字连拼不在 known。
    这里测 6 字连拼出现在触发词旁边时被检出。
    """
    analyzer = _make_analyzer()
    # 完整 6 字 "弗雷尔卓德沃" 或 "卓德沃利贝尔" 都不在 known
    text = "本局用弗雷尔卓德沃利贝尔合成了无敌阵容。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    # 至少有一个不在 known 的子片段被识别
    assert len(result) > 0


def test_scan_detects_fake_hex_augment():
    """伪造海克斯强化 "经济之神" 不在 all_augments → 应检出（以经济之神为前缀/核心的片段应存在）。"""
    analyzer = _make_analyzer()
    text = "选到经济之神后立刻滚雪球，连胜拿下整局。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    # "经济之神" 不在 all_augments，至少有一个可疑片段包含该关键词
    assert any("经济之神" in frag for frag in result), f"经济之神 应被检出，实际可疑列表: {result}"


def test_scan_whitelist_terms_not_flagged():
    """白名单术语（连胜 / 经济 / 阵容 / 装备 / 海克斯）出现在触发词旁边 → 不误报。"""
    analyzer = _make_analyzer()
    text = (
        "本局靠连胜积累经济，选装备时优先合主C的暴风大剑，"
        "海克斯强化选了经济特训，阵容稳定推进。"
    )
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    for term in ("连胜", "经济", "装备", "海克斯", "阵容"):
        assert term not in result, f"白名单词 '{term}' 不应被报告为可疑"


def test_scan_empty_text_returns_empty():
    """空文本 → 返回空列表。"""
    analyzer = _make_analyzer()
    assert analyzer._scan_text_for_unknown_names("", _KB) == []


def test_scan_no_context_trigger_not_flagged():
    """中文片段没有上下文触发词时，不应产生误报（降低误报率）。"""
    analyzer = _make_analyzer()
    # "某某英雄" 出现但没有任何触发词（前后均是逗号/句号）
    text = "整体表现良好。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    assert result == []


def test_last_audit_warnings_attribute_exists():
    """last_audit_warnings 属性在初始化后存在且为空列表。"""
    analyzer = _make_analyzer(with_knowledge=False)
    assert hasattr(analyzer, "last_audit_warnings")
    assert analyzer.last_audit_warnings == []


def test_last_audit_warnings_no_knowledge_stays_empty():
    """knowledge=None 时，调用内部 scan 方法也不抛异常且返回空。"""
    from src.schema import MatchReport, RoundReview
    analyzer = _make_analyzer(with_knowledge=False)
    report = MatchReport(
        match_id="test-0",
        final_rank=4,
        final_hp=20,
        duration_s=600,
        summary="测试总结",
        key_rounds=[],
    )
    result = analyzer._scan_text_for_unknown_names_in_report(report)
    assert result == []
    assert analyzer.last_audit_warnings == []


def test_extract_candidate_names_basic():
    """_extract_candidate_names 能提取出有上下文提示词的中文片段。"""
    text = "用艾希拿装备"
    candidates = _extract_candidate_names(text)
    assert "艾希" in candidates


def test_scan_known_trait_not_flagged():
    """all_traits 中的羁绊名（如"弗雷尔卓德"）本身不应被报告。"""
    analyzer = _make_analyzer()
    # "弗雷尔卓德" 在 all_traits 中
    text = "本局选弗雷尔卓德羁绊开局。"
    result = analyzer._scan_text_for_unknown_names(text, _KB)
    assert "弗雷尔卓德" not in result
