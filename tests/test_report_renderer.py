"""tests/test_report_renderer.py
单元测试 · src/report_renderer.py

覆盖：
1. render_report_html 能生成 > 2000 字符的合法 HTML · 含 <!DOCTYPE html> · 含 match_id
2. 边界：empty key_rounds · 不崩
3. 各 grade (优/可/差) 都有对应 CSS class
4. 多段 summary 换行正常渲染
5. HTML 特殊字符被正确转义
6. 含 rank_tier 时出现在输出中
7. HTML 结构完整（包含 head / body / footer）
8. duration_s 格式化正确
"""
from __future__ import annotations

import time
from typing import Optional

import pytest

from src.schema import MatchReport, RoundReview
from src.report_renderer import render_report_html


# ──────────────────────────────────────────────────────────────────────────────
# 测试固件
# ──────────────────────────────────────────────────────────────────────────────

def _full_report(match_id: str = "TFT-S17-test-001") -> MatchReport:
    """含完整字段的 MatchReport。"""
    return MatchReport(
        match_id=match_id,
        rank_tier="钻石 I",
        final_rank=3,
        final_hp=22,
        duration_s=1980,
        core_comp="法师 6 · 神谕 2 · 卡莎 C",
        key_rounds=[
            RoundReview(
                round="2-1",
                grade="优",
                title="选增强 · 法师之力",
                comment="三选一唯一契合法师 6 的增强，伤害提升 18%。",
                delta="+18% 伤害",
            ),
            RoundReview(
                round="3-2",
                grade="差",
                title="升级后 D 牌不足",
                comment="升 7 级后应继续刷牌至 10g 兜底。",
                delta="-1.5 回合成型",
            ),
            RoundReview(
                round="4-3",
                grade="可",
                title="连败存钱 · HP 告急",
                comment="HP 36 近红线赌连败风险偏高。",
                delta=None,
            ),
        ],
        summary="整体节奏合理 · 3-2 节奏略亏 · 止损及时。\n改进后期望名次从 3.0 → 2.2。",
    )


def _empty_rounds_report() -> MatchReport:
    """key_rounds 为空的 MatchReport。"""
    return MatchReport(
        match_id="TFT-S17-empty-rounds",
        final_rank=5,
        final_hp=0,
        duration_s=600,
        key_rounds=[],
        summary="LLM 未识别出关键回合。",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1 · 基本合法性：长度 · DOCTYPE · match_id
# ──────────────────────────────────────────────────────────────────────────────

class TestBasicValidity:
    def test_output_is_long_enough(self):
        html = render_report_html(_full_report())
        assert len(html) > 2000, f"HTML 太短：{len(html)} 字符"

    def test_doctype_present(self):
        html = render_report_html(_full_report())
        assert "<!DOCTYPE html>" in html

    def test_match_id_in_output(self):
        mid = "TFT-S17-test-001"
        html = render_report_html(_full_report(mid))
        assert mid in html

    def test_html_and_body_tags(self):
        html = render_report_html(_full_report())
        assert "<html" in html
        assert "<body" in html
        assert "</body>" in html
        assert "</html>" in html

    def test_head_contains_charset(self):
        html = render_report_html(_full_report())
        assert 'charset="UTF-8"' in html or "charset=UTF-8" in html

    def test_title_contains_match_id(self):
        html = render_report_html(_full_report("TFT-TITLE-CHECK"))
        assert "TFT-TITLE-CHECK" in html


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2 · 边界：empty key_rounds
# ──────────────────────────────────────────────────────────────────────────────

class TestEmptyRounds:
    def test_empty_rounds_does_not_crash(self):
        report = _empty_rounds_report()
        # 不应抛出任何异常
        html = render_report_html(report)
        assert html  # 非空

    def test_empty_rounds_html_still_valid(self):
        html = render_report_html(_empty_rounds_report())
        assert "<!DOCTYPE html>" in html
        assert len(html) > 1000

    def test_empty_rounds_shows_placeholder(self):
        html = render_report_html(_empty_rounds_report())
        assert "无关键回合" in html

    def test_empty_rounds_match_id_present(self):
        html = render_report_html(_empty_rounds_report())
        assert "TFT-S17-empty-rounds" in html


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3 · grade CSS class
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeCssClasses:
    def _report_with_grade(self, grade: str) -> MatchReport:
        return MatchReport(
            match_id=f"TFT-grade-{grade}",
            final_rank=4,
            final_hp=10,
            duration_s=900,
            key_rounds=[
                RoundReview(
                    round="3-1",
                    grade=grade,  # type: ignore[arg-type]
                    title=f"测试回合 {grade}",
                    comment="测试评论。",
                    delta=None,
                )
            ],
            summary="测试总评。",
        )

    def test_grade_you_has_css_class(self):
        html = render_report_html(self._report_with_grade("优"))
        assert "grade-优" in html

    def test_grade_ke_has_css_class(self):
        html = render_report_html(self._report_with_grade("可"))
        assert "grade-可" in html

    def test_grade_cha_has_css_class(self):
        html = render_report_html(self._report_with_grade("差"))
        assert "grade-差" in html

    def test_all_three_grades_in_one_report(self):
        html = render_report_html(_full_report())
        assert "grade-优" in html
        assert "grade-差" in html
        assert "grade-可" in html


# ──────────────────────────────────────────────────────────────────────────────
# 测试 4 · summary 换行渲染
# ──────────────────────────────────────────────────────────────────────────────

class TestSummaryParagraphs:
    def test_multiline_summary_rendered(self):
        report = MatchReport(
            match_id="TFT-summary-nl",
            final_rank=2,
            final_hp=30,
            duration_s=1800,
            key_rounds=[],
            summary="第一段总评内容。\n第二段改进建议。",
        )
        html = render_report_html(report)
        # 两段文字都应出现
        assert "第一段总评内容。" in html
        assert "第二段改进建议。" in html


# ──────────────────────────────────────────────────────────────────────────────
# 测试 5 · HTML 特殊字符转义
# ──────────────────────────────────────────────────────────────────────────────

class TestHtmlEscaping:
    def test_special_chars_in_match_id_escaped(self):
        # match_id 含 < > & —— 应被转义
        report = MatchReport(
            match_id="TFT-<>&-test",
            final_rank=4,
            final_hp=5,
            duration_s=100,
            key_rounds=[],
            summary="转义测试。",
        )
        html = render_report_html(report)
        # 转义后 < → &lt; 等
        assert "TFT-&lt;&gt;&amp;-test" in html

    def test_special_chars_in_comment_escaped(self):
        report = MatchReport(
            match_id="TFT-escape-comment",
            final_rank=3,
            final_hp=8,
            duration_s=200,
            key_rounds=[
                RoundReview(
                    round="2-1",
                    grade="优",
                    title="测试标题",
                    comment='含<script>alert("xss")</script>的评论',
                    delta=None,
                )
            ],
            summary="安全测试。",
        )
        html = render_report_html(report)
        # script 标签应被转义
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ──────────────────────────────────────────────────────────────────────────────
# 测试 6 · rank_tier 出现在输出中
# ──────────────────────────────────────────────────────────────────────────────

class TestRankTier:
    def test_rank_tier_present_when_set(self):
        html = render_report_html(_full_report())
        assert "钻石 I" in html

    def test_no_rank_tier_does_not_crash(self):
        report = MatchReport(
            match_id="TFT-no-tier",
            final_rank=6,
            final_hp=0,
            duration_s=800,
            rank_tier=None,
            key_rounds=[],
            summary="无段位信息。",
        )
        html = render_report_html(report)
        assert "<!DOCTYPE html>" in html


# ──────────────────────────────────────────────────────────────────────────────
# 测试 7 · footer 结构
# ──────────────────────────────────────────────────────────────────────────────

class TestFooterStructure:
    def test_footer_has_jcc_link(self):
        html = render_report_html(_full_report())
        assert "jcc-replay-analyst" in html

    def test_footer_has_generated_time(self):
        html = render_report_html(_full_report())
        # 生成时间格式 YYYY-MM-DD
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2}", html)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 8 · duration_s 格式化
# ──────────────────────────────────────────────────────────────────────────────

class TestDurationFormatting:
    def test_duration_minutes_seconds(self):
        report = MatchReport(
            match_id="TFT-dur",
            final_rank=4,
            final_hp=5,
            duration_s=33 * 60 + 12,  # 33m12s
            key_rounds=[],
            summary="时长测试。",
        )
        html = render_report_html(report)
        assert "33m" in html
        assert "12s" in html

    def test_duration_with_hours(self):
        report = MatchReport(
            match_id="TFT-dur-long",
            final_rank=1,
            final_hp=80,
            duration_s=3720,  # 1h02m00s
            key_rounds=[],
            summary="超长对局。",
        )
        html = render_report_html(report)
        assert "1h" in html
