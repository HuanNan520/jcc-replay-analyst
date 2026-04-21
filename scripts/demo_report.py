#!/usr/bin/env python3
"""demo_report.py —— 生成一份真实感 MatchReport 并渲染成 HTML 存到 examples/。

优先使用 LocalLLMAnalyzer（vLLM 在端口 8000 上） · 否则 fallback 到写死的 demo Report。

用法：
    python3 scripts/demo_report.py
    python3 scripts/demo_report.py --fallback        # 强制跳 vLLM · 用 fallback
    python3 scripts/demo_report.py --out my.html     # 指定输出路径
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# 保证从 repo 根目录的 src 包能被 import
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.schema import (
    ActiveTrait, BagItem, MatchReport, RoundReview, Unit, WorldState
)
from src.report_renderer import render_report_html

log = logging.getLogger("demo_report")

# ──────────────────────────────────────────────────────────────────────────────
# 合成一局 S17 真实感 WorldState 序列
# ──────────────────────────────────────────────────────────────────────────────

def _build_s17_states() -> list[WorldState]:
    """构造一组有信号的 WorldState 序列 · 覆盖开局 → 成型 → 收尾。"""
    base_ts = time.time() - 2400.0

    def _ts(offset: int) -> float:
        return base_ts + offset

    def _u(name: str, star: int = 1, items: list[str] | None = None) -> Unit:
        return Unit(name=name, star=star, items=items or [], position=None)

    def _t(name: str, count: int, tier: str = "none") -> ActiveTrait:
        return ActiveTrait(name=name, count=count, tier=tier)  # type: ignore[arg-type]

    states: list[WorldState] = []

    # F00 · 1-1 pick 开局
    states.append(WorldState(
        stage="pick", round="1-1", hp=100, gold=0, level=1,
        exp="0/2", board=[], bench=[],
        shop=["安妮", "尼可", "寒冰", "卡莎", "维克多"],
        active_traits=[], augments=[], timestamp=_ts(0),
    ))

    # F01 · 1-2 pve · 补到 2 费
    states.append(WorldState(
        stage="pve", round="1-2", hp=100, gold=2, level=2,
        exp="2/6", board=[_u("安妮"), _u("尼可")],
        bench=[], shop=[], active_traits=[_t("神谕", 2)],
        augments=[], timestamp=_ts(50),
    ))

    # F02 · 2-1 augment · 选增强
    states.append(WorldState(
        stage="augment", round="2-1", hp=98, gold=10, level=3,
        exp="0/10", board=[_u("安妮"), _u("尼可"), _u("法场")],
        bench=[], shop=[],
        active_traits=[_t("神谕", 2), _t("法师", 2)],
        augments=["法师之力", "利息强化", "棱彩法阵"],
        timestamp=_ts(180),
    ))

    # F03 · 2-2 pve · 选了法师之力 · 开始存钱
    states.append(WorldState(
        stage="pve", round="2-2", hp=94, gold=14, level=3,
        exp="4/10", board=[_u("安妮"), _u("尼可"), _u("法场"), _u("维克多")],
        bench=[], shop=[],
        active_traits=[_t("神谕", 2), _t("法师", 3)],
        augments=["法师之力"],
        timestamp=_ts(250),
    ))

    # F04 · 2-5 pve · 经济峰值 · 连胜存到 50g
    states.append(WorldState(
        stage="pve", round="2-5", hp=88, gold=50, level=4,
        exp="2/20", board=[_u("安妮"), _u("尼可"), _u("法场"), _u("维克多"), _u("卡莎", 2)],
        bench=[_u("卡莎")], shop=[],
        active_traits=[_t("神谕", 2), _t("法师", 4)],
        augments=["法师之力"],
        timestamp=_ts(500),
    ))

    # F05 · 3-2 pve · 断崖 · 升 7 后只 D 到 10g
    states.append(WorldState(
        stage="pve", round="3-2", hp=78, gold=8, level=7,
        exp="4/56", board=[_u("安妮"), _u("尼可"), _u("法场"), _u("维克多"), _u("卡莎", 2), _u("里桑德拉"), _u("莫甘娜")],
        bench=[_u("卡莎")], shop=[],
        active_traits=[_t("神谕", 2), _t("法师", 6), _t("虚空", 2)],
        augments=["法师之力"],
        bag=[BagItem(slot=0, name="暴风大剑"), BagItem(slot=1, name="反曲之弓")],
        timestamp=_ts(720),
    ))

    # F06 · 3-5 carousel · 抢装备
    states.append(WorldState(
        stage="carousel", round="3-5", hp=62, gold=20, level=7,
        exp="12/56", board=[_u("安妮"), _u("尼可"), _u("法场"), _u("维克多"), _u("卡莎", 2), _u("里桑德拉"), _u("莫甘娜")],
        bench=[], shop=["寒冰", "金克斯", "维克多", "法场", "蔚"],
        active_traits=[_t("神谕", 2), _t("法师", 6)],
        augments=["法师之力"], timestamp=_ts(900),
    ))

    # F07 · 4-1 pve · 连败存钱 · HP 告急
    states.append(WorldState(
        stage="pve", round="4-1", hp=38, gold=36, level=7,
        exp="18/56", board=[_u("安妮"), _u("尼可"), _u("法场"), _u("维克多"), _u("卡莎", 2), _u("里桑德拉"), _u("莫甘娜")],
        bench=[_u("卡莎")], shop=[],
        active_traits=[_t("神谕", 2), _t("法师", 6), _t("虚空", 2)],
        augments=["法师之力"], timestamp=_ts(1080),
    ))

    # F08 · 5-1 pvp · 止损大 D · 卡莎三星出来了
    states.append(WorldState(
        stage="pvp", round="5-1", hp=18, gold=6, level=8,
        exp="10/80", board=[_u("安妮"), _u("尼可"), _u("法场"), _u("维克多"), _u("卡莎", 3, ["无尽之刃", "巨人腰带", "纳什之牙"]), _u("里桑德拉"), _u("莫甘娜"), _u("维尔戈")],
        bench=[], shop=[],
        active_traits=[_t("神谕", 2), _t("法师", 6), _t("虚空", 3), _t("守护", 2)],
        augments=["法师之力"], timestamp=_ts(1680),
    ))

    # F09 · end · 第 3 名结算
    states.append(WorldState(
        stage="end", round="5-6", hp=14, gold=0, level=8,
        exp="10/80",
        board=[_u("卡莎", 3)],
        bench=[], shop=[], active_traits=[],
        augments=["法师之力"], timestamp=_ts(2400),
    ))

    return states


# ──────────────────────────────────────────────────────────────────────────────
# Fallback 写死 demo Report
# ──────────────────────────────────────────────────────────────────────────────

def _demo_fallback_report() -> MatchReport:
    """写死的 demo MatchReport · vLLM 不可用时用这个产出 HTML。"""
    return MatchReport(
        match_id="TFT-S17-20260422-demo",
        rank_tier="铂金 II",
        final_rank=3,
        final_hp=14,
        duration_s=2400,
        core_comp="法师 6 · 神谕 2 · 卡莎 C",
        key_rounds=[
            RoundReview(
                round="2-1",
                grade="优",
                title="选增强 · 法师之力",
                comment=(
                    "三选一分别是「法师之力」「利息强化」「棱彩法阵」。"
                    "当前手牌已有安妮 + 尼可 + 法场三个 2 费法师底盘，"
                    "「法师之力」与目标阵容法师 6 完全叠加，伤害溢出收益约 +18%。"
                    "「利息强化」在当前存钱路线下收益偏低——存到 50g 利息封顶才 5g/回合，"
                    "而法师之力在每个 PVP 回合均有体现。"
                    "反事实：若选利息强化，整局期望伤害降低 15-20%，最终名次大概率从第 3 滑至第 4-5。"
                    "此步三项（时机/品质/阵容契合）全对，评为优。"
                ),
                delta="+18% 伤害",
            ),
            RoundReview(
                round="3-2",
                grade="差",
                title="升 7 级后 D 牌不足 · 经济断崖",
                comment=(
                    "3-2 升到 7 级后手头 8g，应继续 D 牌到 10g 兜底线寻找卡莎 2 费合体。"
                    "实际操作：升完 7 级仅 D 了 3 次（花约 9g）就停手，"
                    "此时 HP 78 完全不在红线，没有节约的必要。"
                    "错误点在于将「存钱」与「D 牌」混淆——"
                    "存钱是不主动买经验，D 牌找关键棋子是阵容成型的核心操作，两者不冲突。"
                    "此决策直接导致卡莎合体延迟 1.5 回合，"
                    "进而在 3-5 到 4-2 区间多承受了约 12 点血量压力。"
                    "反事实：若 3-2 继续 D 到 10g 兜底，卡莎极大概率在 3-4 前合体，"
                    "后续连败段（4-1~4-3）可提前打出连胜盾，期望名次 2-3。"
                ),
                delta="-1.5 回合成型",
            ),
            RoundReview(
                round="4-1",
                grade="可",
                title="选择连败存钱 · HP 告急",
                comment=(
                    "4-1 手上 36g、HP 38，对手强度已到 4 费卡活跃段。"
                    "此时赌连败存钱属于高风险策略——本赛季 S17 低费卡池相对浅，"
                    "连败 streak 难以稳定维持超过 2 回合。"
                    "HP 38 看似有余量，但 4 费段伤害单回合约 12-16 点，"
                    "3 回合连败即可从 38 跌到 8 以内。"
                    "正确节奏应是 3-5 转换策略——保留约 30g 存量的同时考虑升 8 强化阵容。"
                    "但当前执行已属于「次优但未崩盘」，"
                    "最终因 5-1 止损大 D 卡莎三星及时补救，损失可控。评为可。"
                ),
                delta=None,
            ),
            RoundReview(
                round="5-1",
                grade="优",
                title="止损大 D · 卡莎 3★ 出手",
                comment=(
                    "HP 18 悬崖边，果断将 30g 存量全部 D 出寻找卡莎三星所需的最后一张。"
                    "运营节奏准确：此时继续存钱已无意义（再多挨一个大招就淘汰），"
                    "唯一出路是拿到卡莎三星打出连胜盾止血。"
                    "最终第 3 张卡莎在第 4 次刷新到手，卡莎三星装备「无尽之刃+巨人腰带+纳什之牙」"
                    "完全成型，随后连胜两把从 HP 18 稳住到终局 HP 14 存活第 3 名。"
                    "反事实：若该回合犹豫不决，多挨一把大概率被打出局，名次 5-8。"
                    "关键时刻决策果断，是本局拿回前三的唯一节点。评为优。"
                ),
                delta="+2 名次",
            ),
            RoundReview(
                round="5-6",
                grade="可",
                title="终局 · 法师 6 阵容定型",
                comment=(
                    "最终阵容：卡莎三星（无尽+腰带+纳什）C 位，法师 6、神谕 2、虚空 3、守护 2。"
                    "版本 T1.5 阵容，理论上限第 2-3 名，实际打到第 3，符合正常发挥。"
                    "不足在于装备合成路线：3-2 节点本可以在 D 牌同时顺手合成「饮血之剑」"
                    "而非等到 5-1 才补完装备。装备提前到位后卡莎在 4 费段的单局输出可高出约 25%。"
                    "反事实：若 3-2 装备成型更早，4-1 连败段损血可减少约 8 点，"
                    "止损时机更从容，有较大概率冲击第 2 名。"
                ),
                delta=None,
            ),
        ],
        summary=(
            "整局核心问题是 3-2 升 7 后 D 牌不足，导致卡莎合体延迟 1.5 回合，"
            "进而在 4 费活跃段承受了额外约 12 点被动失血。"
            "5-1 止损操作果断挽回局面，最终第 3 名已是此套法师 6 在当前残局条件下的合理上限。\n"
            "若要改进：① 3-2 升 7 后应继续 D 到 10g 兜底（约多花 9g），"
            "卡莎可提前 1.5 回合合体；② 4-1 不赌连败，保 30g 存量同时评估升 8 时机；"
            "③ 装备合成路线前置，3-3 前完成主 C 核心件。"
            "按上述调整，期望名次从 3.0 上升至 2.0-2.5 区间，约提升 0.5-1 名。"
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# vLLM 尝试 → 生成 MatchReport
# ──────────────────────────────────────────────────────────────────────────────

async def _try_vllm_report(
    base_url: str,
    model: str,
    states: list[WorldState],
) -> MatchReport | None:
    """尝试调 LocalLLMAnalyzer · 失败返回 None。"""
    try:
        from src.llm_analyzer import LocalLLMAnalyzer
        analyzer = LocalLLMAnalyzer(
            base_url=base_url,
            model=model,
            knowledge=None,
            timeout=60.0,
        )
        report = await analyzer.synthesize(states)
        # 简单校验：summary 不是空占位
        if "LLM 未启动" in report.summary or not report.key_rounds:
            return None
        return report
    except Exception as e:
        log.warning("vLLM 调用失败 · %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

async def _async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states = _build_s17_states()

    report: MatchReport | None = None
    used_vllm = False

    if not args.fallback:
        log.info("尝试 vLLM · base_url=%s · model=%s", args.llm_url, args.llm_model)
        report = await _try_vllm_report(args.llm_url, args.llm_model, states)
        if report is not None:
            used_vllm = True
            log.info("vLLM 成功 · match_id=%s · key_rounds=%d", report.match_id, len(report.key_rounds))

    if report is None:
        log.info("使用 fallback demo report")
        report = _demo_fallback_report()

    html_content = render_report_html(report)
    out_path.write_text(html_content, encoding="utf-8")

    size_kb = len(html_content.encode("utf-8")) / 1024
    log.info(
        "HTML 已写出 · %s · %.1f KB · vLLM=%s · match_id=%s",
        out_path, size_kb, used_vllm, report.match_id,
    )

    # 验收输出
    print(f"[demo_report] 输出：{out_path}")
    print(f"[demo_report] 大小：{size_kb:.1f} KB")
    print(f"[demo_report] 来源：{'vLLM 真推理' if used_vllm else 'fallback 写死'}")
    print(f"[demo_report] match_id：{report.match_id}")
    print(f"[demo_report] key_rounds：{len(report.key_rounds)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="demo_report",
        description="生成 S17 真实感 demo 复盘 HTML",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "examples" / "sample_report.html"),
        help="输出 HTML 路径",
    )
    ap.add_argument(
        "--llm-url",
        default="http://localhost:8000/v1",
        help="vLLM 地址",
    )
    ap.add_argument(
        "--llm-model",
        default="Qwen3-VL-4B-FP8",
        help="模型名",
    )
    ap.add_argument(
        "--fallback",
        action="store_true",
        help="跳过 vLLM 直接用 fallback demo",
    )
    args = ap.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
