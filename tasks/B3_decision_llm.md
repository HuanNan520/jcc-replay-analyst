# B3 · 低延迟决策 LLM（6 类）

**分配给**：Claude Opus 4.7（`claude-opus-4-7`）· prompt 工程密集 · 6 套独立 schema + 短 prompt 调优。
**依赖**：无 · 可和 B1 / B4 并行。
**预期工时**：1–1.5 天（含 prompt 迭代）。
**运行时**：**本地 vLLM**（和 A1 同一实例 · 复用 8000 端口 · 模型 Qwen3-VL-4B-FP8）。
**新产品定位中的角色**：实时 tick loop 的**大脑** · 每个决策点触发一次调用 · 目标单次 ≤ 3 秒。

---

## 你是谁

你是被派到 `HuanNan520/jcc-replay-analyst` 执行 B3 的 Claude Opus 4.7。
A1 已经把"合成整局 MatchReport"的 LLM 接入做完了（见 `src/llm_analyzer.py`）· 但那是**整局复盘**风格 · 一次调用要 38 秒、吃 34 帧 WorldState。

**实时场景不能用 A1 的路径** —— 玩家每回合只有 30 秒思考时间 · 每个决策点必须**秒级**给建议。

你的任务：写**六个短 prompt 专家**路由 · 每次调用只针对一个决策点 · 目标 ≤ 3 秒。

## 六类决策点（Decision Kinds）

| kind           | 触发条件                            | LLM 输出核心                         |
|----------------|-------------------------------------|--------------------------------------|
| `augment`      | stage == "augment" · 三选一         | 三张契合度排序 + 推荐 + 理由         |
| `carousel`     | stage == "carousel" · 轮抱          | 棋子优先级排序 + 核心推荐            |
| `shop`         | 商店刷新（任意 pvp 间） · 有待抉择   | 买 / 卖 / 锁定 · 每张卡短评          |
| `level`        | 回合开始 · 金币足够升级             | 升 / 不升 / 留钱 D · 节奏判断        |
| `positioning` | stage == "positioning"               | 主 C 位置 · 诱饵建议 · 对抗最强玩家  |
| `item`         | bag 有 ≥2 个散件可合成              | 合给谁 · 合成啥 · 理由               |

## 目标产物

```python
from src.decision_llm import DecisionLLM, DecisionContext

llm = DecisionLLM(
    base_url="http://localhost:8000/v1",
    model="Qwen3-VL-4B-FP8",
    knowledge=s16_knowledge,
)

ctx = DecisionContext(kind="augment", options=["法师之力", "复利", "攻速强化"], timeout_s=25)
advice = await llm.decide(world_state, ctx)
# advice 是 Advice 的某个子类 · 含 recommendation / reasoning / confidence
```

---

## 具体要做

### 1. 新增 `src/decision_llm.py`

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
    """告诉 LLM 当前是哪类决策 · 路由到对应 prompt。"""
    kind: DecisionKind
    options: list[str] = Field(default_factory=list, description="可选项文本 · augment 三选一 / carousel 棋子名")
    timeout_s: float = Field(default=25.0, description="玩家决策剩余时间 · 可选辅助信息")


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
    actions: list[dict] = Field(..., description="每张卡一个 {slot, action: buy/skip/note}")
    should_lock: bool
    should_reroll: bool


class LevelAdvice(AdviceBase):
    kind: Literal["level"] = "level"
    action: Literal["up", "stay", "roll"]
    hold_gold_above: Optional[int] = Field(None, description="若 action=stay · 建议留多少金币")


class PositioningAdvice(AdviceBase):
    kind: Literal["positioning"] = "positioning"
    main_carry_row: int = Field(..., ge=0, le=3)
    main_carry_col: int = Field(..., ge=0, le=6)
    bait_unit: Optional[str] = Field(None, description="用什么做诱饵")
    notes: list[str] = Field(default_factory=list)


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
    """A3 的 S16Knowledge 已经满足 · duck typing。"""
    def version_context(self) -> str: ...
    def comps_table(self) -> str: ...
    def validate_unit_name(self, name: str) -> bool: ...


# ==================== Per-kind System Prompts ====================

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
    """把 WorldState 压成短文本 · 只给 LLM 当前最关键的信息。"""
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
        timeout: float = 5.0,   # 实时场景紧预算
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.knowledge = knowledge
        self.timeout = timeout

    async def decide(self, ws: WorldState, ctx: DecisionContext) -> Advice:
        """单点决策 · 目标 ≤ 3 秒返回。"""
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
            "max_tokens": 400,   # 实时场景短输出
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
            log.warning("DecisionLLM [%s] 请求失败 · %.2fs · %s", ctx.kind, time.time() - t0, e)
            return self._fallback(ctx, reason=f"LLM 调用失败 · {type(e).__name__}")
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
            log.warning("DecisionLLM [%s] 输出非法 · %s · raw=%r", ctx.kind, e, raw[:300])
            return self._fallback(ctx, reason=f"非法 JSON · {type(e).__name__}")

    def _fallback(self, ctx: DecisionContext, reason: str) -> Advice:
        """LLM 失败时的骨架 Advice · 至少 UI 有东西显示。"""
        cls = ADVICE_CLASSES[ctx.kind]
        # 根据 kind 构造最小合法实例
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

### 2. 单元测试 `tests/test_decision_llm.py`

至少覆盖：
- `_compact_state` 对空 WorldState 不崩
- `DecisionLLM._fallback` 对六类都能构造合法 Advice
- Advice 子类的 pydantic schema 导出包含正确 kind literal
- 一个 mock httpx 返回的 e2e 测试（不连真 LLM）

轻量测试 · 别连真服务。

### 3. 一个 smoke demo

`scripts/demo_decision.py`：

```python
"""对真 vLLM 跑一次 augment 决策 · 看延迟和输出。"""
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
    print(f"耗时: {time.time()-t0:.2f}s")
    print(advice.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 禁止做的事

- 严禁 import anthropic / openai · 就 httpx（和 A1 同风格）
- 不要把 6 个 prompt 塞成一个 god prompt · 拆分有价值（提升稳定性 + 可迭代）
- 不要写"如果 kind=augment elif kind=carousel"的 if/else 链 · 用 dict dispatch（代码里已示范）
- 不要改 `src/llm_analyzer.py` · 那是整局复盘用 · 两套并存
- 不要 coupling knowledge —— knowledge 是 Protocol / Optional · None 时要能降级
- max_tokens 不准超 500（实时场景硬约束）
- 不要改 `schema.py`（DecisionContext / Advice 在你的新文件里定义）

---

## 自验收清单

- [ ] `python -c "from src.decision_llm import DecisionLLM, DecisionContext, AugmentAdvice"` 导入无错
- [ ] `pytest tests/test_decision_llm.py -v` 全绿 · 至少 8 个测试
- [ ] `grep -rn "anthropic\|openai" src/decision_llm.py tests/test_decision_llm.py` 零命中（除非是 "OpenAI 兼容" 这种 comment）
- [ ] vLLM 跑着的话 · `python scripts/demo_decision.py` 能拿到合法 AugmentAdvice · 耗时打印 ≤ 3s
- [ ] 每类 kind 都至少跑过一次 fallback 分支（断网时 decide() 不崩 · 返回骨架 Advice）
- [ ] 和原有 `pytest tests/ -q` 一起跑 · 原 40 个不回归

## 完成后

给用户 ≤ 200 字报告：
- 6 类 prompt 的 token 预算实测（prompt_tokens 范围）
- demo_decision.py 跑出的延迟
- fallback 路径触发时 UI 能显示什么（reasoning 字段内容样例）
- 给 B2 的接口契约：`DecisionLLM(knowledge=...).decide(ws, ctx) -> Advice`

不 git commit。

---

## 参考

- A1 的 `src/llm_analyzer.py:143` 里 httpx + guided_json 调用样例 · 直接抄风格
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- 注意 `extra_body` 在 openai sdk 里叫 `extra_body` · httpx 直接打就是 payload 里加 `extra_body` 字段 · vLLM server 会识别
