"""本地 LLM 分析层 —— 把 WorldState 序列合成一份 MatchReport。

默认路线（非硬约束 · fork 换云自行扩展）：
  - 走本地 vLLM OpenAI 兼容接口（默认 http://localhost:8000/v1）· 自建零成本
  - 不内置 anthropic / openai 等云 SDK 依赖 · 仅 httpx · 要换云自己加一个实现
  - 结构化输出优先用 vLLM guided_json · 降级到 response_format json_object
  - vLLM 内建 automatic prefix caching · system prompt 复用 KV cache · 不需要 client 端
    cache_control（那是 Anthropic 特有）

用法：
    from src.llm_analyzer import LocalLLMAnalyzer
    llm = LocalLLMAnalyzer(base_url="http://localhost:8000/v1",
                           model="Qwen3-VL-8B-FP8",
                           knowledge=s16_knowledge)
    report = await llm.synthesize(states)
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional, Protocol

import httpx

from .schema import MatchReport, RoundReview, WorldState

log = logging.getLogger(__name__)


class KnowledgeProvider(Protocol):
    """A3 会实现 · 本任务只用 Protocol · 可 None。

    src.knowledge.S16Knowledge 已经 duck-typing 兼容 · 直接传即可。
    """
    def version_context(self) -> str: ...
    def comps_table(self) -> str: ...
    def validate_unit_name(self, name: str) -> bool: ...


SYSTEM_HEADER = """你是《金铲铲之战》S16 教练。任务是对一局对局的状态序列做复盘评析。

## 报告质量硬标准
1. 每个 key_round 必须带评级（优/可/差）· 评级依据必须在 comment 里说清楚
2. 每个 key_round 的 comment 必须 **150 字以上** 且包含一个**反事实** —— "如果这步不这样 · 期望第 X 名"
3. summary 必须 **200 字以上** · 含**量化影响**估计 —— "本局若改 XX · 期望排名从 4.0 → 2.X"
4. 不准输出 TODO / placeholder / 骨架 / 省略符 / 空字符串
5. 评语必须基于状态序列的数据 · 不许凭空编造未发生的事件
6. key_rounds **必须 3-6 条** · 选局势转折点（大病 / 选增强 / 连败 / 关键 D 牌）
7. 即使输入信号稀疏（例如 HP/金币均为 0 · 阵容为空）· 也必须按 schema 输出占位
   分析 · 用通用 TFT 教学建议填满字数要求 · 绝对不许返回 key_rounds=[] 或空 summary
"""


SCHEMA_DESCRIPTION = """## 输出 schema

严格按 MatchReport JSON 输出 · 字段：
- match_id (str) · final_rank (1-8 整数) · final_hp (0-100 整数) · duration_s (秒) ·
  core_comp (str|null) · rank_tier (str|null · 例"钻石 I")
- key_rounds: list of {round, grade, title, comment, delta}
  - round 格式 "3-2" 之类
  - grade 只能是 "优" / "可" / "差"
  - title 30 字以内 · 点出本回合做的主要动作
  - comment 150-300 字 · 含评级依据 + 反事实
  - delta 可为 null 或短量化描述 · 例 "+18% 伤害" / "-1.5 回合成型"
- summary: str · 200-400 字 · 整局总评 + 量化反事实

严格只输出合法 JSON · 不准 markdown 代码块包裹 · 不准前后废话。
"""


# 白名单：常见 TFT/金铲铲通用术语，不当英雄名对待
_WHITELIST_TERMS: set[str] = {
    # TFT 核心术语
    "阵容", "羁绊", "装备", "海克斯", "主C", "副C", "经济", "连胜", "连败",
    "强化", "增强", "转型", "D牌", "过渡", "节奏", "卡位", "满级", "刷新",
    "前期", "中期", "后期", "开局", "收尾", "大病", "连损", "血线", "优势",
    "劣势", "胜局", "败局", "吃鸡", "稳健", "激进", "保守", "扩展", "压血",
    "小兵", "对战", "摆位", "棋盘", "备战", "候补", "三星", "二星", "一星",
    "最终", "结算", "赛季", "版本", "强势", "弱势", "主流", "冷门", "特色",
    "大后期", "小后期", "极限",
    # 常见中文短语（普通动作/叙述）· 防止被误判为专有名词
    "本局", "这局", "此局", "全局", "整局",
    "若能", "若有", "如果", "假如", "否则",
    "成功", "失败", "推进", "进攻", "防守",
    "无敌", "强力", "效果", "影响", "操作",
    "关键", "重要", "核心", "基础", "基本",
    "稳定", "稳健", "激进", "选择", "考虑",
    "开始", "结束", "完成", "达成", "实现",
    "使用", "采用", "选用", "优先", "尽量",
    "此时", "当时", "此刻", "时机", "时间",
    "位置", "地位", "角色", "功能", "作用",
    "连接", "合成", "拼接", "组合", "配合",
    "无法", "能够", "可以", "应该", "必须",
    "提升", "降低", "增加", "减少", "改变",
}

# 上下文提示词：前后有这些词时，中间片段更可能是专有名词（英雄/装备名）
_CONTEXT_TRIGGERS: set[str] = {
    "选", "到", "用", "合", "买", "卖", "推", "打",
    "升", "出", "拿", "换", "带", "配", "开", "组",
    "《", "【", "「", "（", "‘", "“",
}

# 中文连续片段提取正则
_CJK_BLOCK_RE = re.compile(r"[一-鿿]+")
# 英雄名样式：纯中文 2-6 字
_HERO_NAME_RE = re.compile(r"^[一-鿿]{2,6}$")


def _extract_candidate_names(text: str) -> list[str]:
    """从文本中提取 2-6 字中文片段候选（上下文提示词触发版）。

    策略：
    - 遍历每个连续中文块，对 2-6 字滑动窗口内的片段做检测
    - 片段本身不能以触发词起始或结尾（触发词应在片段外部邻接）
    - 片段邻接字符中有触发词才收集
    """
    candidates: list[str] = []
    for block_match in _CJK_BLOCK_RE.finditer(text):
        block = block_match.group()
        block_start = block_match.start()
        blen = len(block)
        for size in range(2, 7):           # 2-6 字窗口
            for offset in range(blen - size + 1):
                frag = block[offset: offset + size]
                if not _HERO_NAME_RE.match(frag):
                    continue
                # 片段不应以触发词起始或结尾（触发词是名称的上下文，不是名称本身）
                if frag[0] in _CONTEXT_TRIGGERS or frag[-1] in _CONTEXT_TRIGGERS:
                    continue
                abs_start = block_start + offset
                abs_end = abs_start + size
                pre_char = text[abs_start - 1] if abs_start > 0 else ""
                post_char = text[abs_end] if abs_end < len(text) else ""
                if pre_char in _CONTEXT_TRIGGERS or post_char in _CONTEXT_TRIGGERS:
                    candidates.append(frag)
    return candidates


class LocalLLMAnalyzer:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-VL-8B-FP8",
        knowledge: Optional[KnowledgeProvider] = None,
        timeout: float = 90.0,
        api_key: str = "EMPTY",
        use_guided_json: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.knowledge = knowledge
        self.timeout = timeout
        self.api_key = api_key
        self.use_guided_json = use_guided_json
        self.last_audit_warnings: list[str] = []

    async def synthesize(self, states: List[WorldState]) -> MatchReport:
        if not states:
            return _empty_report()

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._compact_states(states)},
        ]

        base_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 3500,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # 方案 1 · vLLM guided_json · 最稳 · 采样层强制 JSON schema
        # 方案 2（降级）· response_format json_object · 只保证合法 JSON
        attempts = []
        if self.use_guided_json:
            attempts.append(("guided_json",
                             {**base_payload,
                              "extra_body": {"guided_json": MatchReport.model_json_schema()}}))
        attempts.append(("json_object",
                         {**base_payload,
                          "response_format": {"type": "json_object"}}))

        body = None
        dt = 0.0
        mode_used = None
        last_err: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for mode, payload in attempts:
                t0 = time.time()
                try:
                    r = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    r.raise_for_status()
                    body = r.json()
                    dt = time.time() - t0
                    mode_used = mode
                    break
                except Exception as e:
                    dt = time.time() - t0
                    last_err = e
                    log.warning("LLM 调用失败 · mode=%s · %.1fs · err=%s", mode, dt, e)
                    continue

        if body is None:
            log.error("LLM 所有模式均失败 · 最后错误 %s · 降级空报告", last_err)
            return _empty_report(states)

        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {}) or {}
        log.info(
            "LLM synthesize · %.1fs · mode=%s · prompt=%d · completion=%d · chars=%d",
            dt, mode_used,
            usage.get("prompt_tokens", -1),
            usage.get("completion_tokens", -1),
            len(raw),
        )

        raw_clean = _strip_markdown_fence(raw)
        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError as e:
            log.warning("LLM 非法 JSON · 降级 mock · err=%s · head=%r", e, raw_clean[:200])
            return _empty_report(states)

        report = _coerce_match_report(data, states)
        if self.knowledge is not None:
            self._audit_hallucinations(report)
            warnings = self._scan_text_for_unknown_names_in_report(report)
            if warnings:
                self.last_audit_warnings = warnings
                log.warning("LLM 全文扫描可疑专有名词 · %s", warnings)
        return report

    def _build_system_prompt(self) -> str:
        parts = [SYSTEM_HEADER]
        if self.knowledge is not None:
            try:
                parts.append("## S16 版本知识\n" + self.knowledge.version_context())
                parts.append("## 强势阵容表\n" + self.knowledge.comps_table())
            except Exception as e:
                log.warning("knowledge provider 调用失败 · 降级通用 · %s", e)
                parts.append(_GENERIC_KNOWLEDGE_NOTE)
        else:
            parts.append(_GENERIC_KNOWLEDGE_NOTE)
        parts.append(SCHEMA_DESCRIPTION)
        return "\n\n".join(parts)

    def _compact_states(self, states: List[WorldState]) -> str:
        lines = []
        signal_strength = 0  # 非零字段计数 · 用来提示 LLM 输入是否稀疏
        for i, ws in enumerate(states):
            board = ",".join(f"{u.name}★{u.star}" for u in ws.board[:8])
            traits = ",".join(f"{t.name}×{t.count}" for t in ws.active_traits[:4])
            augs = ",".join(ws.augments) if ws.augments else "-"
            if ws.hp or ws.gold or ws.board or ws.active_traits or ws.augments:
                signal_strength += 1
            lines.append(
                f"[F{i:02d}] round={ws.round} stage={ws.stage} hp={ws.hp} "
                f"gold={ws.gold} lvl={ws.level} traits={traits or '-'} "
                f"board={board or '-'} augs={augs}"
            )
        duration = int(states[-1].timestamp - states[0].timestamp) if len(states) > 1 else 0
        header = (
            f"本局共 {len(states)} 关键帧 · 时长 {duration}s · "
            f"最终 HP {states[-1].hp} · 最终等级 {states[-1].level}。\n\n"
        )
        if signal_strength == 0:
            header += (
                "【注意】本序列识别层信号极弱 · 所有帧 HP/金币/阵容均为空。"
                "请仍按 schema 输出 3-6 条 key_round · 结合回合数/时长给出**通用 TFT 教学建议**"
                "（常见开局曲线 · 2-1/3-2/4-1 节奏要点 · 选增强/经济/D 牌通用原则）·"
                "summary 给一段通用复盘 · 把反事实落在「若数据可见时一般做法」上。\n\n"
            )
        return header + "\n".join(lines)

    def _scan_text_for_unknown_names(self, text: str, knowledge) -> list[str]:
        """tokenization 级别全文扫描 · 检测未知专有名词（不依赖引号）。

        策略：
        1. 维护 known = all_units | all_traits | all_items | all_augments
        2. 用滑动窗口提取 2-6 字中文连续片段（仅上下文触发词邻接的片段）
        3. 片段不在 known + 不在白名单 + 不是任何 known 词的子串 → 可疑候选
        4. 去重后，仅保留不被其他可疑候选包含的最长形式（减少重复碎片噪声）
        """
        known: set[str] = set()
        for attr in ("all_units", "all_traits", "all_items", "all_augments"):
            val = getattr(knowledge, attr, None)
            if isinstance(val, set):
                known |= val

        raw_candidates: list[str] = []
        seen: set[str] = set()
        whitelist_all = _WHITELIST_TERMS | known
        for candidate in _extract_candidate_names(text):
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in whitelist_all:
                continue
            # 过滤：片段是某个已知词/白名单词的子串（防止已知词局部碎片误报）
            if any(candidate in kw for kw in whitelist_all):
                continue
            # 过滤：片段完全由某个已知词/白名单词 + 紧接后缀构成（如"弗雷尔卓德羁"）
            # 判据：片段以某个 ≥2 字的已知词开头 或 结尾，且已知词占片段长度 >50%
            skip = False
            for kw in whitelist_all:
                if len(kw) < 2:
                    continue
                if (candidate.startswith(kw) or candidate.endswith(kw)) and len(kw) / len(candidate) > 0.5:
                    skip = True
                    break
            if skip:
                continue
            raw_candidates.append(candidate)

        # 仅保留不被其他候选包含的最长形式（去掉噪声子串）
        suspicious: list[str] = [
            c for c in raw_candidates
            if not any(c != other and c in other for other in raw_candidates)
        ]
        return suspicious

    def _scan_text_for_unknown_names_in_report(self, report: MatchReport) -> list[str]:
        """对 report 的 summary / key_round.comment / core_comp 做全文扫描。"""
        if self.knowledge is None:
            return []
        texts: list[str] = []
        if report.summary:
            texts.append(report.summary)
        if report.core_comp:
            texts.append(report.core_comp)
        for rr in report.key_rounds:
            if rr.comment:
                texts.append(rr.comment)
        combined = "".join(texts)
        return self._scan_text_for_unknown_names(combined, self.knowledge)

    def _audit_hallucinations(self, report: MatchReport) -> None:
        """对 LLM 提到的英雄名跑 knowledge 校验 · 仅 log warn · 不改文本。"""
        suspicious: set[str] = set()
        pattern = re.compile(r'"([一-鿿]{2,6})"')
        texts: list[str] = []
        for rr in report.key_rounds:
            if rr.comment:
                texts.append(rr.comment)
            if rr.title:
                texts.append(rr.title)
        if report.summary:
            texts.append(report.summary)
        for text in texts:
            for name in pattern.findall(text):
                try:
                    if not self.knowledge.validate_unit_name(name):
                        suspicious.add(name)
                except Exception:
                    pass
        if suspicious:
            log.warning("LLM hallucinate 可疑英雄名 · %s", sorted(suspicious))


_GENERIC_KNOWLEDGE_NOTE = (
    "## 版本知识\n（未注入 · 请用通用 TFT 机制 · 避免具体英雄名 hallucinate · "
    "用\"主 C\"/\"副 C\"/\"坦克\"/\"辅助\"等通用角色称谓）"
)


def _strip_markdown_fence(text: str) -> str:
    """LLM 偶尔会违规包 ```json ... ``` · 容错剥掉。"""
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s.split("```", 2)[1]
    if s.lstrip().startswith("json"):
        s = s.split("\n", 1)[1] if "\n" in s else s[4:]
    return s.rstrip("`").strip()


def _coerce_match_report(data: dict, states: List[WorldState]) -> MatchReport:
    """把 LLM 松散输出强行规范成合法 MatchReport · 参考 vlm_client._coerce_world_state。"""

    def _s(v, default: str = "") -> str:
        return str(v) if v is not None else default

    def _i(v, default: int = 0, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
        try:
            r = int(v)
        except (TypeError, ValueError):
            try:
                r = int(float(v))
            except (TypeError, ValueError):
                return default
        if lo is not None and r < lo:
            r = lo
        if hi is not None and r > hi:
            r = hi
        return r

    def _opt_s(v) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    last = states[-1] if states else None
    first = states[0] if states else None
    fallback_hp = last.hp if last is not None else 0
    fallback_duration = (
        int(last.timestamp - first.timestamp)
        if (last is not None and first is not None and len(states) > 1)
        else 0
    )
    fallback_match_id = f"TFT-S16-{int(time.time())}"

    def _coerce_round(rr: dict) -> Optional[RoundReview]:
        try:
            grade = _s(rr.get("grade", "可"))
            if grade not in {"优", "可", "差"}:
                grade = "可"
            round_str = _s(rr.get("round", "?-?")) or "?-?"
            title = _s(rr.get("title", "")) or "(未命名回合)"
            comment = _s(rr.get("comment", "")) or "(LLM 未给出评语)"
            delta = _opt_s(rr.get("delta"))
            return RoundReview(
                round=round_str,
                grade=grade,  # type: ignore[arg-type]
                title=title,
                comment=comment,
                delta=delta,
            )
        except Exception as e:
            log.debug("round coerce fail · %s · %s", e, rr)
            return None

    raw_rounds = data.get("key_rounds") or []
    if not isinstance(raw_rounds, list):
        raw_rounds = []
    key_rounds = [r for r in (_coerce_round(x) for x in raw_rounds if isinstance(x, dict)) if r]

    summary = _s(data.get("summary", "")).strip()
    if not summary:
        summary = "（LLM 未给出总评 · 字段缺失）"

    return MatchReport(
        match_id=_s(data.get("match_id"), fallback_match_id) or fallback_match_id,
        rank_tier=_opt_s(data.get("rank_tier")),
        final_rank=_i(data.get("final_rank", 4), 4, 1, 8),
        final_hp=_i(data.get("final_hp", fallback_hp), fallback_hp, 0, 100),
        duration_s=_i(data.get("duration_s", fallback_duration), fallback_duration, 0),
        core_comp=_opt_s(data.get("core_comp")),
        key_rounds=key_rounds,
        summary=summary,
    )


def _empty_report(states: Optional[List[WorldState]] = None) -> MatchReport:
    states = states or []
    last_hp = states[-1].hp if states else 0
    duration = (
        int(states[-1].timestamp - states[0].timestamp)
        if len(states) > 1 else 0
    )
    return MatchReport(
        match_id=f"TFT-S16-{int(time.time())}",
        final_rank=4,
        final_hp=last_hp,
        duration_s=duration,
        key_rounds=[],
        summary="（LLM 未启动 / 解析失败 · 无法生成分析）",
    )
