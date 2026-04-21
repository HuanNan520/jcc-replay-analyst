"""B3 · DecisionLLM 单元测试。

不连真 LLM · 用 respx mock httpx 或直接单元测 pure function。
覆盖：
  - `_compact_state` 对空 WorldState 不崩
  - `DecisionLLM._fallback` 对六类 kind 都能构造合法 Advice
  - Advice 子类的 pydantic schema 导出包含正确 kind literal
  - prompt builder 字典 dispatch 完整
  - 每类 kind 的 fallback reasoning 带 `（降级 · xxx）` 前缀
  - mock httpx 的 e2e · 走 guided_json 正常路径
  - mock httpx 的 e2e · 超时降级
  - mock httpx · 非法 JSON 响应降级
  - knowledge=None 时 prompt 能构造（不崩）
  - knowledge 提供时 prompt 注入 version_context()
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from src.decision_llm import (
    ADVICE_CLASSES,
    PROMPT_BUILDERS,
    Advice,
    AdviceBase,
    AugmentAdvice,
    CarouselAdvice,
    DecisionContext,
    DecisionLLM,
    ItemAdvice,
    LevelAdvice,
    PositioningAdvice,
    ShopAdvice,
    _compact_state,
)
from src.schema import ActiveTrait, BagItem, Unit, WorldState


# ==================== Fixtures ====================

def _blank_ws() -> WorldState:
    """空 WorldState · 所有计数字段 0。"""
    return WorldState(
        stage="unknown",
        round="0-0",
        hp=0,
        gold=0,
        level=1,
        exp="0/0",
        timestamp=0.0,
    )


def _rich_ws() -> WorldState:
    """带少量数据的 WorldState · 测压缩格式。"""
    return WorldState(
        stage="augment",
        round="2-1",
        hp=82,
        gold=28,
        level=5,
        exp="12/20",
        board=[
            Unit(name="安妮", star=2, items=[]),
            Unit(name="阿狸", star=1, items=[]),
        ],
        bench=[Unit(name="剑圣", star=1)],
        bag=[BagItem(slot=0, name="暴风大剑"), BagItem(slot=1, name="反曲之弓")],
        shop=["亚索", "卢锡安", "寒冰", "瑟庄妮", "李青"],
        active_traits=[ActiveTrait(name="法师", count=2, tier="bronze")],
        augments=["复利"],
        timestamp=time.time(),
    )


class _FakeKnowledge:
    """最小 KnowledgeProvider 实现 · 测 knowledge 注入。"""

    def __init__(self, ctx_text: str = "版本:S17 测试", comps_text: str = "| 阵容 | 评分 |"):
        self._ctx = ctx_text
        self._comps = comps_text

    def version_context(self) -> str:
        return self._ctx

    def comps_table(self) -> str:
        return self._comps

    def validate_unit_name(self, name: str) -> bool:
        return name in {"安妮", "阿狸", "剑圣", "亚索"}


# ==================== compact_state ====================

def test_compact_state_on_blank_does_not_crash():
    ws = _blank_ws()
    out = _compact_state(ws)
    assert isinstance(out, str)
    assert "round=0-0" in out
    assert "traits: -" in out
    assert "board: -" in out
    assert "bag: -" in out
    assert "shop: -" in out
    assert "augments so far: -" in out


def test_compact_state_on_rich_includes_all_sections():
    ws = _rich_ws()
    out = _compact_state(ws)
    assert "hp=82" in out
    assert "gold=28" in out
    assert "安妮★2" in out
    assert "阿狸★1" in out
    assert "剑圣" in out
    assert "法师×2" in out
    assert "暴风大剑" in out
    assert "亚索" in out
    assert "复利" in out


# ==================== Advice schema literal kind ====================

@pytest.mark.parametrize(
    "cls,expected_kind",
    [
        (AugmentAdvice, "augment"),
        (CarouselAdvice, "carousel"),
        (ShopAdvice, "shop"),
        (LevelAdvice, "level"),
        (PositioningAdvice, "positioning"),
        (ItemAdvice, "item"),
    ],
)
def test_advice_schema_has_correct_kind_literal(cls: type[AdviceBase], expected_kind: str):
    schema = cls.model_json_schema()
    # kind 应为 literal const == expected_kind
    kind_prop = schema["properties"]["kind"]
    # pydantic v2 的 Literal 可能用 const 或 enum
    const = kind_prop.get("const")
    enum = kind_prop.get("enum")
    assert const == expected_kind or enum == [expected_kind], (
        f"{cls.__name__} kind literal 不符 · schema={kind_prop}"
    )


def test_prompt_builders_cover_all_kinds():
    assert set(PROMPT_BUILDERS.keys()) == {
        "augment", "carousel", "shop", "level", "positioning", "item",
    }
    assert set(ADVICE_CLASSES.keys()) == set(PROMPT_BUILDERS.keys())


# ==================== Fallback: six kinds ====================

@pytest.mark.parametrize(
    "kind,opts",
    [
        ("augment", ["法师之力", "复利", "攻速强化"]),
        ("augment", []),  # 空 options 也要能降级
        ("augment", ["单个"]),  # 不足 3 个 · 应 pad
        ("carousel", ["亚索", "卢锡安"]),
        ("carousel", []),
        ("shop", []),
        ("level", []),
        ("positioning", []),
        ("item", []),
    ],
)
def test_fallback_for_all_kinds_produces_valid_advice(kind: str, opts: list[str]):
    llm = DecisionLLM()
    ctx = DecisionContext(kind=kind, options=opts)  # type: ignore[arg-type]
    advice = llm._fallback(ctx, reason=f"单元测试 · {kind}")

    # 基础字段检查
    assert advice.kind == kind
    assert advice.confidence == 0.0
    assert "降级" in advice.reasoning
    assert f"单元测试 · {kind}" in advice.reasoning

    # 类型正确
    assert isinstance(advice, ADVICE_CLASSES[kind])

    # kind-specific 合法性
    if isinstance(advice, AugmentAdvice):
        assert len(advice.ranked) == 3  # 始终 pad/trunc 到 3
    if isinstance(advice, CarouselAdvice):
        assert isinstance(advice.priority, list)
    if isinstance(advice, ShopAdvice):
        assert advice.should_lock is False
        assert advice.should_reroll is False
        assert advice.actions == []
    if isinstance(advice, LevelAdvice):
        assert advice.action == "stay"
    if isinstance(advice, PositioningAdvice):
        assert 0 <= advice.main_carry_row <= 3
        assert 0 <= advice.main_carry_col <= 6
    if isinstance(advice, ItemAdvice):
        assert len(advice.combine) == 2


# ==================== Prompt builders ====================

def test_all_prompts_include_sys_header_and_quality():
    ctx = DecisionContext(kind="augment", options=["A", "B", "C"])
    for kind, builder in PROMPT_BUILDERS.items():
        ctx_k = DecisionContext(kind=kind, options=ctx.options)  # type: ignore[arg-type]
        text = builder(ctx_k, None)
        assert "金铲铲" in text
        assert "S17" in text, f"{kind} prompt 里应显式声明 S17"
        assert "## 输出 schema" in text
        assert "## 输出质量硬标准" in text


def test_prompt_without_knowledge_uses_fallback_text():
    ctx = DecisionContext(kind="augment", options=["法师之力", "复利", "攻速强化"])
    text = PROMPT_BUILDERS["augment"](ctx, None)
    assert "无知识库" in text


def test_prompt_with_knowledge_injects_version_context():
    ctx = DecisionContext(kind="augment", options=["法师之力", "复利", "攻速强化"])
    fake = _FakeKnowledge(ctx_text="版本 S17-TEST 内容")
    text = PROMPT_BUILDERS["augment"](ctx, fake)
    assert "版本 S17-TEST 内容" in text


def test_carousel_prompt_uses_comps_table():
    ctx = DecisionContext(kind="carousel", options=["安妮", "亚索"])
    fake = _FakeKnowledge(comps_text="| 超级阵容 | S |")
    text = PROMPT_BUILDERS["carousel"](ctx, fake)
    assert "超级阵容" in text


# ==================== DecisionLLM.decide · mock httpx ====================

class _MockTransport(httpx.AsyncBaseTransport):
    """返回预设响应的 httpx transport。"""

    def __init__(self, response_body: dict | str, status: int = 200, raise_exc: Exception | None = None):
        self.response_body = response_body
        self.status = status
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            self.calls.append({"url": str(request.url), "json": json.loads(request.content)})
        except Exception:
            self.calls.append({"url": str(request.url)})
        if self.raise_exc is not None:
            raise self.raise_exc
        if isinstance(self.response_body, dict):
            body = json.dumps(self.response_body).encode()
        else:
            body = self.response_body.encode()
        return httpx.Response(
            status_code=self.status,
            headers={"content-type": "application/json"},
            content=body,
        )


def _patched_client_factory(transport: _MockTransport):
    """返回一个 AsyncClient 构造 callable · 用来 monkeypatch httpx.AsyncClient。"""
    original = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    return factory


def test_decide_happy_path(monkeypatch):
    """mock vLLM 返回合法 AugmentAdvice JSON · decide 应成功解析。"""
    ws = _rich_ws()
    ctx = DecisionContext(
        kind="augment",
        options=["法师之力", "复利", "攻速强化"],
    )

    llm_reply = {
        "kind": "augment",
        "reasoning": "当前阵容法师 2 羁绊已激活 · 法师之力能直接叠加技能强度 · 远优于其余两个经济类 / 通用攻速。",
        "confidence": 0.82,
        "ranked": ["法师之力", "复利", "攻速强化"],
        "recommendation": "法师之力",
    }

    transport = _MockTransport({
        "choices": [
            {"message": {"content": json.dumps(llm_reply)}}
        ],
        "usage": {"prompt_tokens": 350, "completion_tokens": 120},
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM()
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, AugmentAdvice)
    assert advice.recommendation == "法师之力"
    assert advice.confidence == pytest.approx(0.82)
    assert advice.ranked[0] == "法师之力"

    # 请求体应包含 guided_json schema · 并指向 /chat/completions
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert "extra_body" in call["json"]
    assert "guided_json" in call["json"]["extra_body"]
    # max_tokens 硬约束 ≤ 500
    assert call["json"]["max_tokens"] <= 500


def test_decide_timeout_falls_back(monkeypatch):
    """httpx 抛 TimeoutException · decide 应返回 fallback · 不抛。"""
    ws = _blank_ws()
    ctx = DecisionContext(kind="level", options=[])

    transport = _MockTransport(
        response_body={},
        raise_exc=httpx.TimeoutException("simulated timeout"),
    )
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM(timeout=0.1)
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, LevelAdvice)
    assert advice.confidence == 0.0
    assert "降级" in advice.reasoning
    assert "TimeoutException" in advice.reasoning or "LLM 调用失败" in advice.reasoning


def test_decide_invalid_json_falls_back(monkeypatch):
    """LLM 吐非法 JSON · decide 应走 fallback · 不抛。"""
    ws = _blank_ws()
    ctx = DecisionContext(kind="shop", options=[])

    transport = _MockTransport({
        "choices": [
            {"message": {"content": "这是自然语言 不是 JSON"}}
        ],
        "usage": {},
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM()
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, ShopAdvice)
    assert advice.confidence == 0.0
    assert "降级" in advice.reasoning


def test_decide_schema_validation_fail_falls_back(monkeypatch):
    """LLM 吐合法 JSON 但缺字段 · pydantic 校验失败 · 应 fallback。"""
    ws = _blank_ws()
    ctx = DecisionContext(kind="item", options=[])

    transport = _MockTransport({
        "choices": [
            {"message": {"content": json.dumps({"kind": "item", "reasoning": "缺字段"})}}
        ],
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM()
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, ItemAdvice)
    assert advice.confidence == 0.0
    assert "降级" in advice.reasoning


def test_decide_uses_knowledge_version_context(monkeypatch):
    """knowledge 非 None 时 · decide 应把 version_context 塞进 system prompt。"""
    ws = _rich_ws()
    ctx = DecisionContext(kind="augment", options=["A", "B", "C"])

    fake_k = _FakeKnowledge(ctx_text="注入的版本字符串 XYZ")

    llm_reply = {
        "kind": "augment",
        "reasoning": "ok" * 20,
        "confidence": 0.5,
        "ranked": ["A", "B", "C"],
        "recommendation": "A",
    }
    transport = _MockTransport({
        "choices": [{"message": {"content": json.dumps(llm_reply)}}],
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM(knowledge=fake_k)
    asyncio.run(llm.decide(ws, ctx))

    call = transport.calls[0]
    sys_msg = call["json"]["messages"][0]["content"]
    assert "注入的版本字符串 XYZ" in sys_msg


# ==================== guided_json 可开关 ====================

def test_decide_without_guided_json(monkeypatch):
    """use_guided_json=False · 请求不带 extra_body。"""
    ws = _blank_ws()
    ctx = DecisionContext(kind="level", options=[])

    llm_reply = {
        "kind": "level",
        "reasoning": "blood low · reroll",
        "confidence": 0.6,
        "action": "roll",
    }
    transport = _MockTransport({
        "choices": [{"message": {"content": json.dumps(llm_reply)}}],
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM(use_guided_json=False)
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, LevelAdvice)
    assert advice.action == "roll"
    call = transport.calls[0]
    assert "extra_body" not in call["json"]


# ==================== DecisionContext 自身 ====================

def test_decision_context_default_timeout():
    ctx = DecisionContext(kind="augment", options=["A", "B", "C"])
    assert ctx.timeout_s == 25.0


def test_decision_context_accepts_all_kinds():
    for kind in ("augment", "carousel", "shop", "level", "positioning", "item"):
        DecisionContext(kind=kind)  # type: ignore[arg-type]


# ==================== PositioningAdvice coerce 容错 ====================

@pytest.mark.parametrize(
    "row_in,col_in,expected_row,expected_col,desc",
    [
        (3, 2, 3, 2, "normal int"),
        ("3", "6", 3, 6, "string digits → int"),
        (2.9, 5.1, 2, 5, "float → int (truncated)"),
        (10, 99, 3, 6, "out-of-range high → clamped to max"),
        (-5, -1, 0, 0, "negative → clamped to 0"),
        ("后排", "右下", 3, 3, "non-numeric string → default fallback"),
        ("3.0", "6.0", 3, 6, "float-string → int"),
    ],
)
def test_positioning_advice_coerce_row_col(row_in, col_in, expected_row, expected_col, desc):
    """PositioningAdvice 对 main_carry_row/col 的 coerce 容错验证。"""
    advice = PositioningAdvice(
        kind="positioning",
        reasoning="对位测试 " * 10,
        confidence=0.7,
        main_carry_row=row_in,
        main_carry_col=col_in,
    )
    assert advice.main_carry_row == expected_row, f"[{desc}] row mismatch"
    assert advice.main_carry_col == expected_col, f"[{desc}] col mismatch"
    # 最终值始终在有效边界内
    assert 0 <= advice.main_carry_row <= 3, f"[{desc}] row out of bounds"
    assert 0 <= advice.main_carry_col <= 6, f"[{desc}] col out of bounds"


def test_positioning_advice_coerce_does_not_affect_other_fields():
    """coerce 只影响 row/col · 不改动 reasoning/confidence/bait_unit/notes。"""
    advice = PositioningAdvice(
        kind="positioning",
        reasoning="对位分析：刺客可跳后排 · 建议主C放右侧角落 · 盖伦前排吸引仇恨。",
        confidence=0.85,
        main_carry_row="2",  # string → 2
        main_carry_col="5",  # string → 5
        bait_unit="盖伦",
        notes=["前排放盖伦", "主C放右侧"],
    )
    assert advice.reasoning.startswith("对位分析")
    assert advice.confidence == pytest.approx(0.85)
    assert advice.bait_unit == "盖伦"
    assert advice.notes == ["前排放盖伦", "主C放右侧"]
    assert advice.main_carry_row == 2
    assert advice.main_carry_col == 5


def test_positioning_coerce_with_mock_llm_string_output(monkeypatch):
    """模拟 LLM 吐字符串数字 · decide 走正常路径而非 fallback。"""
    import asyncio
    import json

    ws = _blank_ws()
    ctx = DecisionContext(kind="positioning", options=[])

    # LLM 吐字符串形式的坐标（常见 bad output）
    llm_reply = {
        "kind": "positioning",
        "reasoning": "对手有刺客 · 主C建议放后排角落 · 盖伦前排吸引火力 · 安妮后排输出更安全。",
        "confidence": 0.75,
        "main_carry_row": "3",   # 字符串 → coerce → 3
        "main_carry_col": "0",   # 字符串 → coerce → 0
        "bait_unit": None,
        "notes": [],
    }

    transport = _MockTransport({
        "choices": [{"message": {"content": json.dumps(llm_reply)}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 80},
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM()
    advice = asyncio.run(llm.decide(ws, ctx))

    # 关键：不走 fallback · confidence > 0
    assert isinstance(advice, PositioningAdvice)
    assert advice.confidence == pytest.approx(0.75), "不应走 fallback"
    assert advice.main_carry_row == 3
    assert advice.main_carry_col == 0


def test_positioning_coerce_with_mock_llm_outofrange_output(monkeypatch):
    """模拟 LLM 吐越界数值 · coerce 后 clamp · 不走 fallback。"""
    import asyncio
    import json

    ws = _blank_ws()
    ctx = DecisionContext(kind="positioning", options=[])

    # LLM 吐越界值
    llm_reply = {
        "kind": "positioning",
        "reasoning": "对手有巨魔 · 主C建议放最深后排右侧 · 拉开距离避免被冲脸 · 前排盖伦挡线。",
        "confidence": 0.68,
        "main_carry_row": 7,    # 越界 → clamp → 3
        "main_carry_col": 10,   # 越界 → clamp → 6
        "bait_unit": "盖伦",
        "notes": ["前排顶线"],
    }

    transport = _MockTransport({
        "choices": [{"message": {"content": json.dumps(llm_reply)}}],
        "usage": {"prompt_tokens": 210, "completion_tokens": 85},
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM()
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, PositioningAdvice)
    assert advice.confidence == pytest.approx(0.68), "不应走 fallback"
    assert 0 <= advice.main_carry_row <= 3
    assert 0 <= advice.main_carry_col <= 6
    assert advice.main_carry_row == 3   # clamped
    assert advice.main_carry_col == 6   # clamped


# ==================== Zero hallucinated imports ====================

def test_decision_llm_source_has_no_openai_or_anthropic_sdk_import():
    """硬约束 · decision_llm.py 不准 import openai / anthropic SDK。"""
    import inspect

    import src.decision_llm as mod

    src_text = inspect.getsource(mod)
    # 允许 "OpenAI 兼容" 这种 comment · 禁止真的 import
    for banned in ("import openai", "from openai", "import anthropic", "from anthropic"):
        assert banned not in src_text, f"禁止 {banned}"
