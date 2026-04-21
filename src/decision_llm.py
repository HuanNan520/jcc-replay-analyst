"""低延迟决策 LLM —— 六类决策点的短 prompt 专家。

和 `llm_analyzer.py` 并存但职责正交：
  - `llm_analyzer.LocalLLMAnalyzer.synthesize()` 做整局复盘 · ≥30 秒预算 · 吃整串 WorldState
  - `DecisionLLM.decide()` 做单点决策 · ≤3 秒预算 · 吃单帧 WorldState

每个 decision kind 有独立 prompt · 使用字典 dispatch（不写 if/elif 链）。
外部结构化输出走 vLLM `guided_json` · httpx 直连 · 不依赖 openai/anthropic SDK。

运行时路线：
  - 默认本地 vLLM OpenAI 兼容接口 · 端口 8000 · 模型 Qwen3-VL-4B-FP8
  - knowledge 可选（A3 的 S16Knowledge / S17Knowledge 鸭子类型兼容）· None 时降级到通用规则
  - LLM 失败 / 输出非 JSON 时返回对应 Advice 子类的骨架实例 · 永不抛异常给 UI

注意：prompt 里统一用 "S17"（国服当前赛季）· 即使知识库是 S16 回退也要让 LLM 知道
当前讨论的是 S17 环境 —— knowledge 里自带版本号前缀会覆盖这个。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Literal, Optional, Protocol, Union

import httpx
from pydantic import BaseModel, Field, field_validator

from .schema import WorldState

log = logging.getLogger(__name__)


# ==================== Decision Context ====================

DecisionKind = Literal["augment", "carousel", "shop", "level", "positioning", "item"]


class DecisionContext(BaseModel):
    """告诉 LLM 当前是哪类决策 · 路由到对应 prompt。"""
    kind: DecisionKind
    options: list[str] = Field(
        default_factory=list,
        description="可选项文本 · augment 三选一 / carousel 棋子名 / shop 5 张卡",
    )
    timeout_s: float = Field(
        default=25.0,
        description="玩家决策剩余时间 · 提示 LLM 不要给过于复杂的推理",
    )


# ==================== Advice Output Schemas ====================

class AdviceBase(BaseModel):
    kind: DecisionKind
    reasoning: str = Field(..., description="简短理由 · 50-150 字")
    confidence: float = Field(..., ge=0, le=1)


class AugmentAdvice(AdviceBase):
    kind: Literal["augment"] = "augment"
    ranked: list[str] = Field(..., description="三个选项从最优到最差 · 含契合度标签")
    recommendation: str = Field(..., description="直接说选哪个 · 'A' / 'B' / 'C' 或名字")


class CarouselAdvice(AdviceBase):
    kind: Literal["carousel"] = "carousel"
    priority: list[str] = Field(..., description="棋子从最优到最差")
    recommendation: str


class ShopAdvice(AdviceBase):
    kind: Literal["shop"] = "shop"
    actions: list[dict] = Field(
        ...,
        description="每张卡一个 {slot:int, action:'buy'|'skip'|'note', note:str}",
    )
    should_lock: bool
    should_reroll: bool


class LevelAdvice(AdviceBase):
    kind: Literal["level"] = "level"
    action: Literal["up", "stay", "roll"]
    hold_gold_above: Optional[int] = Field(
        None, description="若 action=stay · 建议留多少金币(利息门槛)"
    )


class PositioningAdvice(AdviceBase):
    kind: Literal["positioning"] = "positioning"
    main_carry_row: int = Field(..., ge=0, le=3)
    main_carry_col: int = Field(..., ge=0, le=6)
    bait_unit: Optional[str] = Field(None, description="用什么做诱饵")
    notes: list[str] = Field(default_factory=list)

    @field_validator("main_carry_row", mode="before")
    @classmethod
    def _coerce_row(cls, v: object) -> int:
        """容错：将字符串/浮点 coerce 成 int · 越界后 clamp 到 [0, 3]。"""
        try:
            v = int(float(str(v)))
        except (ValueError, TypeError):
            v = 3  # 默认后排
        return max(0, min(3, v))

    @field_validator("main_carry_col", mode="before")
    @classmethod
    def _coerce_col(cls, v: object) -> int:
        """容错：将字符串/浮点 coerce 成 int · 越界后 clamp 到 [0, 6]。"""
        try:
            v = int(float(str(v)))
        except (ValueError, TypeError):
            v = 3  # 默认中间列
        return max(0, min(6, v))


class ItemAdvice(AdviceBase):
    kind: Literal["item"] = "item"
    target_unit: str
    combine: list[str] = Field(..., description="两个组件合成目标装备名")
    hold_for_later: list[str] = Field(default_factory=list)


Advice = Union[
    AugmentAdvice, CarouselAdvice, ShopAdvice,
    LevelAdvice, PositioningAdvice, ItemAdvice,
]


# ==================== Knowledge Provider Protocol ====================

class KnowledgeProvider(Protocol):
    """A3 的 S16Knowledge / S17Knowledge 已经满足 · duck typing。"""
    def version_context(self) -> str: ...
    def comps_table(self) -> str: ...
    def validate_unit_name(self, name: str) -> bool: ...


# ==================== Per-kind System Prompts ====================

SYS_HEADER = (
    "你是《金铲铲之战》S17（国服当前赛季）实战教练。当前在对局中 · "
    "玩家要做一个决策 · 你给出最优建议。"
)

SYS_QUALITY = """## 输出质量硬标准
1. 只针对当前决策 · 不扯整局复盘
2. reasoning 50-150 字 · 给出评估依据 · 不空话
3. confidence ∈ [0,1] · 犹豫就压低
4. 严格输出 JSON · 不准 markdown 不准解释"""


def _build_augment_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    kb = k.version_context() if k else "（无知识库 · 用通用 TFT 机制判断强化契合度）"
    return f"""{SYS_HEADER}

## 当前决策 · 选强化符文（三选一）
候选强化: {ctx.options}

## 版本知识
{kb}

{SYS_QUALITY}

## 输出 schema (AugmentAdvice)
{{
  "kind": "augment",
  "reasoning": "为何这么排 · 结合当前阵容/经济/回合",
  "confidence": 0.0-1.0,
  "ranked": ["最强选项原文", "中间选项原文", "最弱选项原文"],
  "recommendation": "直接说选哪个 · 用选项原文或 A/B/C"
}}"""


def _build_carousel_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    comps = k.comps_table() if k else "（无知识库 · 按通用强势阵容判断）"
    return f"""{SYS_HEADER}

## 当前决策 · 轮抱（共享转盘）选秀
候选棋子（从中挑一个)：{ctx.options}

## S17 版本阵容梯队
{comps}

{SYS_QUALITY}

## 输出 schema (CarouselAdvice)
{{
  "kind": "carousel",
  "reasoning": "结合手里阵容指向 · 说明为什么这个优先级",
  "confidence": 0.0-1.0,
  "priority": ["最优", "次优", "..."],
  "recommendation": "第一优先抢的棋子名"
}}"""


def _build_shop_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    kb = k.version_context() if k else "（无知识库 · 用通用 TFT 经济/买卡原则）"
    return f"""{SYS_HEADER}

## 当前决策 · 商店决策
当前商店刷出最多 5 张卡。结合手牌、血线、金币、回合、成型进度 · 给每张卡买/跳的建议 · 再决定是否锁店或刷新。

## 版本知识
{kb}

{SYS_QUALITY}

## 输出 schema (ShopAdvice)
{{
  "kind": "shop",
  "reasoning": "经济状态 + 核心缺什么 · 简要说",
  "confidence": 0.0-1.0,
  "actions": [
    {{"slot": 0, "action": "buy", "note": "3 费主 C 凑 2 星"}},
    {{"slot": 1, "action": "skip", "note": "与阵容不契合"}}
  ],
  "should_lock": true,
  "should_reroll": false
}}"""


def _build_level_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 升不升人口
关键判断点 · 评估血量、金币、经济复利（10/20/30/40/50 档利息）、阵容成型节奏。
- up = 现在升一级
- stay = 吃利息不升不 D
- roll = 当前等级全 D 搜本级强卡

{SYS_QUALITY}

## 输出 schema (LevelAdvice)
{{
  "kind": "level",
  "reasoning": "按血量/金币/节奏说依据 · 70-130 字",
  "confidence": 0.0-1.0,
  "action": "up",
  "hold_gold_above": 50
}}"""


def _build_positioning_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 棋盘摆位
评估对手可能的切入型（刺客跳后排 / 巨魔冲脸 / 直线 AOE） · 给出主 C 最优坐标 · 可指定一个前排做诱饵。

坐标系：row 0 是最前排（靠近对手）· row 3 是自家后排 · col 0-6 从左到右。

{SYS_QUALITY}

## 输出 schema (PositioningAdvice)
{{
  "kind": "positioning",
  "reasoning": "对位威胁分析 + 为什么主 C 在这格",
  "confidence": 0.0-1.0,
  "main_carry_row": 3,
  "main_carry_col": 0,
  "bait_unit": "诱饵英雄名或 null",
  "notes": ["辅助位提示 1", "坦克位提示 2"]
}}"""


def _build_item_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 装备合成
bag 里有至少 2 个散件可合成 · 决定合给哪个单位 · 合成什么装备 · 哪些散件暂不合留后用。

{SYS_QUALITY}

## 输出 schema (ItemAdvice)
{{
  "kind": "item",
  "reasoning": "主 C 缺什么 · 为什么先合这件",
  "confidence": 0.0-1.0,
  "target_unit": "合给谁的中文英雄名",
  "combine": ["组件 1 中文名", "组件 2 中文名"],
  "hold_for_later": ["暂不合的散件"]
}}"""


PROMPT_BUILDERS = {
    "augment": _build_augment_prompt,
    "carousel": _build_carousel_prompt,
    "shop": _build_shop_prompt,
    "level": _build_level_prompt,
    "positioning": _build_positioning_prompt,
    "item": _build_item_prompt,
}


ADVICE_CLASSES: dict[str, type[AdviceBase]] = {
    "augment": AugmentAdvice,
    "carousel": CarouselAdvice,
    "shop": ShopAdvice,
    "level": LevelAdvice,
    "positioning": PositioningAdvice,
    "item": ItemAdvice,
}


# ==================== DecisionLLM Main Class ====================

def _compact_state(ws: WorldState) -> str:
    """把 WorldState 压成短文本 · 只给 LLM 当前最关键的信息。"""
    board = ", ".join(f"{u.name}★{u.star}" for u in ws.board[:10])
    bench = ", ".join(u.name for u in ws.bench[:9])
    traits = ", ".join(f"{t.name}×{t.count}" for t in ws.active_traits[:6])
    bag = ", ".join(f"slot{b.slot}:{b.name}" for b in ws.bag[:9])
    shop = ", ".join(ws.shop[:5])
    return (
        f"round={ws.round} stage={ws.stage} hp={ws.hp} gold={ws.gold} "
        f"lvl={ws.level} exp={ws.exp}\n"
        f"traits: {traits or '-'}\n"
        f"board: {board or '-'}\n"
        f"bench: {bench or '-'}\n"
        f"bag: {bag or '-'}\n"
        f"shop: {shop or '-'}\n"
        f"augments so far: {', '.join(ws.augments) or '-'}\n"
    )


class DecisionLLM:
    """单点决策 LLM · 实时 tick loop 里每个决策点触发一次。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-VL-4B-FP8",
        knowledge: Optional[KnowledgeProvider] = None,
        timeout: float = 5.0,  # 实时场景紧预算
        api_key: str = "EMPTY",
        use_guided_json: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.knowledge = knowledge
        self.timeout = timeout
        self.api_key = api_key
        self.use_guided_json = use_guided_json

    async def decide(self, ws: WorldState, ctx: DecisionContext) -> Advice:
        """单点决策 · 目标 ≤ 3 秒返回。

        永不抛异常 —— LLM 任何失败（超时 / 网络 / 非法 JSON / schema 不符）
        都回 `_fallback` 骨架 Advice · UI 至少有东西显示。
        """
        if ctx.kind not in PROMPT_BUILDERS:
            log.warning("DecisionLLM 未知 kind: %s · 返回 fallback", ctx.kind)
            return self._fallback(ctx, reason=f"未知 kind={ctx.kind}")

        system_prompt = PROMPT_BUILDERS[ctx.kind](ctx, self.knowledge)
        user_prompt = _compact_state(ws)
        advice_cls = ADVICE_CLASSES[ctx.kind]

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,  # 实时场景短输出
        }
        if self.use_guided_json:
            payload["extra_body"] = {
                "guided_json": advice_cls.model_json_schema(),
            }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                body = r.json()
        except Exception as e:
            log.warning(
                "DecisionLLM [%s] 请求失败 · %.2fs · %s",
                ctx.kind, time.time() - t0, e,
            )
            return self._fallback(ctx, reason=f"LLM 调用失败 · {type(e).__name__}")
        dt = time.time() - t0

        try:
            raw = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            log.warning("DecisionLLM [%s] 响应体结构异常 · %s", ctx.kind, e)
            return self._fallback(ctx, reason="响应体结构异常")

        usage = body.get("usage", {}) or {}
        log.info(
            "DecisionLLM [%s] · %.2fs · prompt=%d · completion=%d",
            ctx.kind,
            dt,
            usage.get("prompt_tokens", -1),
            usage.get("completion_tokens", -1),
        )

        try:
            data = json.loads(raw)
            return advice_cls.model_validate(data)
        except Exception as e:
            log.warning(
                "DecisionLLM [%s] 输出非法 · %s · raw=%r",
                ctx.kind, e, raw[:300],
            )
            return self._fallback(ctx, reason=f"非法 JSON · {type(e).__name__}")

    def _fallback(self, ctx: DecisionContext, reason: str) -> Advice:
        """LLM 失败时的骨架 Advice · 至少 UI 有东西显示。

        reasoning 字段带 `（降级 · xxx）` 前缀 · UI 可据此标灰。
        """
        base = {
            "kind": ctx.kind,
            "reasoning": f"（降级 · {reason}）",
            "confidence": 0.0,
        }
        if ctx.kind == "augment":
            opts = ctx.options or ["?", "?", "?"]
            # pad / truncate 到 3 个
            if len(opts) < 3:
                opts = opts + ["?"] * (3 - len(opts))
            return AugmentAdvice(**base, ranked=opts[:3], recommendation="—")
        if ctx.kind == "carousel":
            return CarouselAdvice(
                **base,
                priority=list(ctx.options) or ["?"],
                recommendation=(ctx.options[0] if ctx.options else "—"),
            )
        if ctx.kind == "shop":
            return ShopAdvice(
                **base, actions=[], should_lock=False, should_reroll=False,
            )
        if ctx.kind == "level":
            return LevelAdvice(**base, action="stay", hold_gold_above=None)
        if ctx.kind == "positioning":
            return PositioningAdvice(
                **base, main_carry_row=3, main_carry_col=3, bait_unit=None, notes=[],
            )
        if ctx.kind == "item":
            return ItemAdvice(
                **base, target_unit="?", combine=["?", "?"], hold_for_later=[],
            )
        raise ValueError(f"unknown kind: {ctx.kind}")


__all__ = [
    "DecisionKind",
    "DecisionContext",
    "AdviceBase",
    "AugmentAdvice",
    "CarouselAdvice",
    "ShopAdvice",
    "LevelAdvice",
    "PositioningAdvice",
    "ItemAdvice",
    "Advice",
    "KnowledgeProvider",
    "DecisionLLM",
    "PROMPT_BUILDERS",
    "ADVICE_CLASSES",
]
