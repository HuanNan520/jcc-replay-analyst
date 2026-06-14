# B3 · low-latency decision LLM (6 kinds)

**Assigned to**: Claude Opus 4.7 (`claude-opus-4-7`) · prompt-engineering heavy · 6 independent schemas + short-prompt tuning.
**Dependencies**: none · can run in parallel with B1 / B4.
**Estimated effort**: 1–1.5 days (including prompt iteration).
**Runtime**: **local vLLM** (same instance as A1 · reuses port 8000 · model Qwen3-VL-4B-FP8).
**Role in the new product positioning**: the **brain** of the real-time tick loop · fires once per decision point · target ≤ 3 seconds per call.

---

## Who you are

You are the Claude Opus 4.7 dispatched to `HuanNan520/jcc-replay-analyst` to execute B3.
A1 already finished the LLM integration for "synthesize a whole-match MatchReport" (see `src/llm_analyzer.py`) · but that is the **full-match replay** style · one call takes 38 seconds and consumes 34 frames of WorldState.

**The real-time scenario can't use A1's path** — the player has only 30 seconds of thinking time per round · every decision point must give advice in **seconds**.

Your task: write a router of **six short-prompt experts** · each call targets one decision point · target ≤ 3 seconds.

## Six decision kinds

| kind           | trigger condition                   | LLM output core                       |
|----------------|-------------------------------------|---------------------------------------|
| `augment`      | stage == "augment" · three-pick     | three options ranked by fit + recommendation + reasoning |
| `carousel`     | stage == "carousel" · carousel grab | champion priority ranking + core recommendation |
| `shop`         | shop refresh (any pvp interval) · decision pending | buy / sell / lock · short note per card |
| `level`        | round start · enough gold to level  | up / stay / hold-and-roll · tempo call |
| `positioning`  | stage == "positioning"              | main-carry position · bait advice · counter the strongest player |
| `item`         | bag has ≥2 components to combine     | give to whom · combine into what · reasoning |

## Target deliverable

```python
from src.decision_llm import DecisionLLM, DecisionContext

llm = DecisionLLM(
    base_url="http://localhost:8000/v1",
    model="Qwen3-VL-4B-FP8",
    knowledge=s16_knowledge,
)

ctx = DecisionContext(kind="augment", options=["法师之力", "复利", "攻速强化"], timeout_s=25)
advice = await llm.decide(world_state, ctx)
# advice is one subclass of Advice · contains recommendation / reasoning / confidence
```

---

## What to do

### 1. Add `src/decision_llm.py`

```python
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Literal, Optional, Protocol, Union

import httpx
from pydantic import BaseModel, Field

from .schema import WorldState

log = logging.getLogger(__name__)


# ==================== Decision Context ====================

DecisionKind = Literal["augment", "carousel", "shop", "level", "positioning", "item"]


class DecisionContext(BaseModel):
    """Tells the LLM which kind of decision this is · routes to the matching prompt."""
    kind: DecisionKind
    options: list[str] = Field(default_factory=list, description="option text · augment three-pick / carousel champion names")
    timeout_s: float = Field(default=25.0, description="player's remaining decision time · optional auxiliary info")


# ==================== Advice Output Schemas ====================

class AdviceBase(BaseModel):
    kind: DecisionKind
    reasoning: str = Field(..., description="short reasoning · 50-150 chars")
    confidence: float = Field(..., ge=0, le=1)


class AugmentAdvice(AdviceBase):
    kind: Literal["augment"] = "augment"
    ranked: list[str] = Field(..., description="the three options from best to worst · with a fit label")
    recommendation: str = Field(..., description="say which one directly · 'A' / 'B' / 'C' or the name")


class CarouselAdvice(AdviceBase):
    kind: Literal["carousel"] = "carousel"
    priority: list[str] = Field(..., description="champions from best to worst")
    recommendation: str


class ShopAdvice(AdviceBase):
    kind: Literal["shop"] = "shop"
    actions: list[dict] = Field(..., description="one {slot, action: buy/skip/note} per card")
    should_lock: bool
    should_reroll: bool


class LevelAdvice(AdviceBase):
    kind: Literal["level"] = "level"
    action: Literal["up", "stay", "roll"]
    hold_gold_above: Optional[int] = Field(None, description="if action=stay · suggested gold to keep")


class PositioningAdvice(AdviceBase):
    kind: Literal["positioning"] = "positioning"
    main_carry_row: int = Field(..., ge=0, le=3)
    main_carry_col: int = Field(..., ge=0, le=6)
    bait_unit: Optional[str] = Field(None, description="what to use as bait")
    notes: list[str] = Field(default_factory=list)


class ItemAdvice(AdviceBase):
    kind: Literal["item"] = "item"
    target_unit: str
    combine: list[str] = Field(..., description="the two components combine into the target item name")
    hold_for_later: list[str] = Field(default_factory=list)


Advice = Union[
    AugmentAdvice, CarouselAdvice, ShopAdvice,
    LevelAdvice, PositioningAdvice, ItemAdvice,
]


# ==================== Knowledge Provider Protocol ====================

class KnowledgeProvider(Protocol):
    """A3's S16Knowledge already satisfies this · duck typing."""
    def version_context(self) -> str: ...
    def comps_table(self) -> str: ...
    def validate_unit_name(self, name: str) -> bool: ...


# ==================== Per-kind System Prompts ====================
# NOTE: the prompt bodies below are the Chinese instructions fed to the LLM for the China-server game.
# They are functional model input · kept in Chinese on purpose (do not translate).

SYS_HEADER = "你是《金铲铲之战》S16 实战教练。当前在对局中 · 玩家要做一个决策 · 你给出最优建议。"

SYS_QUALITY = """## 输出质量硬标准
1. 只针对当前决策 · 不扯整局复盘
2. reasoning 50-150 字 · 给出评估依据 · 不空话
3. confidence ∈ [0,1] · 犹豫就压低
4. 严格输出 JSON · 不准 markdown 不准解释
"""


def _build_augment_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    kb = k.version_context() if k else "（无知识库 · 用通用 TFT 机制）"
    return f"""{SYS_HEADER}

## 当前决策 · 选增强（三选一）
选项: {ctx.options}

## 版本知识
{kb}

{SYS_QUALITY}

## 输出 schema (AugmentAdvice)
{{
  "kind": "augment",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "ranked": ["最强", "中间", "最弱"],
  "recommendation": "直接说选哪个"
}}
"""


def _build_carousel_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 轮抱选秀
候选棋子: {ctx.options}

## 版本阵容梯队
{k.comps_table() if k else "（无知识库 · 按通用强势阵容判断）"}

{SYS_QUALITY}

## 输出 schema (CarouselAdvice)
{{
  "kind": "carousel",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "priority": ["最优", "次优", ...],
  "recommendation": "..."
}}
"""


def _build_shop_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    kb = k.version_context() if k else "（无知识库）"
    return f"""{SYS_HEADER}

## 当前决策 · 商店决策
当前刷出 5 张卡 · 结合手牌、血线、金币、回合判断买卖锁刷。

## 版本知识
{kb}

{SYS_QUALITY}

## 输出 schema (ShopAdvice)
{{
  "kind": "shop",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "actions": [{{"slot": 0, "action": "buy/skip", "note": "简短"}}],
  "should_lock": true/false,
  "should_reroll": true/false
}}
"""


def _build_level_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 升不升人口
关键判断点 · 评估血量、金币、经济复利、阵容成型节奏。

{SYS_QUALITY}

## 输出 schema (LevelAdvice)
{{
  "kind": "level",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "action": "up" or "stay" or "roll",
  "hold_gold_above": 50 (若 stay · 建议留钱数)
}}
"""


def _build_positioning_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 棋盘摆位
评估对手可能的切入型（刺客/跳入/直线AOE） · 给出主 C 最优位置。
坐标系：row 0 是最前排（靠近对手）· row 3 是自家后排 · col 0-6 从左到右。

{SYS_QUALITY}

## 输出 schema (PositioningAdvice)
{{
  "kind": "positioning",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "main_carry_row": 0-3,
  "main_carry_col": 0-6,
  "bait_unit": "诱饵英雄名或 null",
  "notes": ["其他建议"]
}}
"""


def _build_item_prompt(ctx: DecisionContext, k: Optional[KnowledgeProvider]) -> str:
    return f"""{SYS_HEADER}

## 当前决策 · 装备合成
bag 里有散件 · 决定合给谁 · 合成什么装备。

{SYS_QUALITY}

## 输出 schema (ItemAdvice)
{{
  "kind": "item",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "target_unit": "英雄名",
  "combine": ["组件1", "组件2"],
  "hold_for_later": ["暂不合的散件"]
}}
"""


PROMPT_BUILDERS = {
    "augment": _build_augment_prompt,
    "carousel": _build_carousel_prompt,
    "shop": _build_shop_prompt,
    "level": _build_level_prompt,
    "positioning": _build_positioning_prompt,
    "item": _build_item_prompt,
}


ADVICE_CLASSES = {
    "augment": AugmentAdvice,
    "carousel": CarouselAdvice,
    "shop": ShopAdvice,
    "level": LevelAdvice,
    "positioning": PositioningAdvice,
    "item": ItemAdvice,
}


# ==================== DecisionLLM Main Class ====================

def _compact_state(ws: WorldState) -> str:
    """Compresses WorldState into short text · gives the LLM only the most critical current info."""
    board = ", ".join(f"{u.name}★{u.star}" for u in ws.board[:10])
    bench = ", ".join(u.name for u in ws.bench[:9])
    traits = ", ".join(f"{t.name}×{t.count}" for t in ws.active_traits[:6])
    bag = ", ".join(f"slot{b.slot}:{b.name}" for b in ws.bag[:9])
    shop = ", ".join(ws.shop[:5])
    return (
        f"round={ws.round} stage={ws.stage} hp={ws.hp} gold={ws.gold} lvl={ws.level} exp={ws.exp}\n"
        f"traits: {traits or '-'}\n"
        f"board: {board or '-'}\n"
        f"bench: {bench or '-'}\n"
        f"bag: {bag or '-'}\n"
        f"shop: {shop or '-'}\n"
        f"augments so far: {', '.join(ws.augments) or '-'}\n"
    )


class DecisionLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-VL-4B-FP8",
        knowledge: Optional[KnowledgeProvider] = None,
        timeout: float = 5.0,   # tight budget for the real-time scenario
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.knowledge = knowledge
        self.timeout = timeout

    async def decide(self, ws: WorldState, ctx: DecisionContext) -> Advice:
        """Single-point decision · target return ≤ 3 seconds."""
        system_prompt = PROMPT_BUILDERS[ctx.kind](ctx, self.knowledge)
        user_prompt = _compact_state(ws)

        advice_cls = ADVICE_CLASSES[ctx.kind]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,   # short output for the real-time scenario
            "extra_body": {
                "guided_json": advice_cls.model_json_schema(),
            },
        }

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                body = r.json()
        except Exception as e:
            log.warning("DecisionLLM [%s] request failed · %.2fs · %s", ctx.kind, time.time() - t0, e)
            return self._fallback(ctx, reason=f"LLM call failed · {type(e).__name__}")
        dt = time.time() - t0

        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        log.info(
            "DecisionLLM [%s] · %.2fs · prompt=%d · completion=%d",
            ctx.kind, dt, usage.get("prompt_tokens", -1), usage.get("completion_tokens", -1),
        )

        try:
            data = json.loads(raw)
            return advice_cls.model_validate(data)
        except Exception as e:
            log.warning("DecisionLLM [%s] invalid output · %s · raw=%r", ctx.kind, e, raw[:300])
            return self._fallback(ctx, reason=f"invalid JSON · {type(e).__name__}")

    def _fallback(self, ctx: DecisionContext, reason: str) -> Advice:
        """Skeleton Advice when the LLM fails · so the UI at least has something to show."""
        cls = ADVICE_CLASSES[ctx.kind]
        # build the minimal valid instance per kind
        # NOTE: the reasoning string is shown in the overlay (Chinese UI) · kept as functional output
        base = {"kind": ctx.kind, "reasoning": f"（降级 · {reason}）", "confidence": 0.0}
        if ctx.kind == "augment":
            return AugmentAdvice(**base, ranked=ctx.options or ["?", "?", "?"], recommendation="—")
        if ctx.kind == "carousel":
            return CarouselAdvice(**base, priority=ctx.options or [], recommendation="—")
        if ctx.kind == "shop":
            return ShopAdvice(**base, actions=[], should_lock=False, should_reroll=False)
        if ctx.kind == "level":
            return LevelAdvice(**base, action="stay")
        if ctx.kind == "positioning":
            return PositioningAdvice(**base, main_carry_row=3, main_carry_col=3)
        if ctx.kind == "item":
            return ItemAdvice(**base, target_unit="?", combine=["?", "?"])
        raise ValueError(f"unknown kind: {ctx.kind}")
```

### 2. Unit test `tests/test_decision_llm.py`

Cover at least:
- `_compact_state` doesn't crash on an empty WorldState
- `DecisionLLM._fallback` can build a valid Advice for all six kinds
- The pydantic schema export of each Advice subclass contains the correct kind literal
- One e2e test with a mocked httpx response (no real LLM connection)

Keep it lightweight · don't hit the real service.

### 3. A smoke demo

`scripts/demo_decision.py`:

```python
"""Run one augment decision against the real vLLM · check latency and output."""
import asyncio
from src.decision_llm import DecisionLLM, DecisionContext
from src.knowledge import load_s16_knowledge
from src.schema import WorldState
import time

async def main():
    ws = WorldState(
        stage="augment", round="2-1", hp=82, gold=28, level=5, exp="12/20",
        augments=[], timestamp=time.time(),
    )
    ctx = DecisionContext(
        kind="augment",
        options=["法师之力", "复利", "攻速强化"],
        timeout_s=25,
    )
    llm = DecisionLLM(knowledge=load_s16_knowledge())
    t0 = time.time()
    advice = await llm.decide(ws, ctx)
    print(f"elapsed: {time.time()-t0:.2f}s")
    print(advice.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## What not to do

- Absolutely no `import anthropic` / `openai` · just httpx (same style as A1)
- Don't cram the 6 prompts into one god prompt · splitting has value (more stability + easier iteration)
- Don't write an "if kind=augment elif kind=carousel" chain · use dict dispatch (already demonstrated in the code)
- Don't modify `src/llm_analyzer.py` · that's for full-match replay · the two coexist
- Don't couple knowledge — knowledge is a Protocol / Optional · must degrade gracefully when None
- max_tokens must not exceed 500 (hard constraint for the real-time scenario)
- Don't modify `schema.py` (DecisionContext / Advice are defined in your new file)

---

## Self-acceptance checklist

- [ ] `python -c "from src.decision_llm import DecisionLLM, DecisionContext, AugmentAdvice"` imports without error
- [ ] `pytest tests/test_decision_llm.py -v` all green · at least 8 tests
- [ ] `grep -rn "anthropic\|openai" src/decision_llm.py tests/test_decision_llm.py` zero hits (unless it's a comment like "OpenAI-compatible")
- [ ] If vLLM is running · `python scripts/demo_decision.py` gets a valid AugmentAdvice · printed latency ≤ 3s
- [ ] Every kind has exercised the fallback branch at least once (offline, decide() doesn't crash · returns a skeleton Advice)
- [ ] Run together with the existing `pytest tests/ -q` · the original 40 don't regress

## After completion

Give the user a ≤ 200-word report:
- Measured token budget for the 6 prompt kinds (prompt_tokens range)
- The latency demo_decision.py produced
- What the UI can show when the fallback path fires (sample of the reasoning field content)
- The interface contract for B2: `DecisionLLM(knowledge=...).decide(ws, ctx) -> Advice`

No git commit.

---

## References

- The httpx + guided_json call example at A1's `src/llm_analyzer.py:143` · copy the style directly
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- Note `extra_body` is called `extra_body` in the openai sdk · hitting httpx directly just means adding an `extra_body` field in the payload · the vLLM server recognizes it
