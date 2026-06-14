"""B3 · DecisionLLM unit tests.

No real LLM connection; mock httpx with respx or unit-test pure functions directly.
Coverage:
  - `_compact_state` does not crash on an empty WorldState
  - `DecisionLLM._fallback` builds a valid Advice for all six kinds
  - the pydantic schema exported by each Advice subclass carries the correct kind literal
  - the prompt-builder dict dispatch is complete
  - the fallback reasoning for each kind carries the `（降级 · xxx）` (fallback) prefix
  - mock-httpx e2e on the normal guided_json path
  - mock-httpx e2e with timeout fallback
  - mock-httpx with an invalid JSON response falling back
  - the prompt can be built when knowledge=None (no crash)
  - when knowledge is provided, the prompt injects version_context()
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
    """Empty WorldState, all count fields 0."""
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
    """WorldState with a little data, for testing the compact format."""
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
    """Minimal KnowledgeProvider implementation, for testing knowledge injection."""

    def __init__(self, ctx_text: str = "version:S17 test", comps_text: str = "| comp | score |"):
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
    # kind should be a literal const == expected_kind
    kind_prop = schema["properties"]["kind"]
    # pydantic v2's Literal may use const or enum
    const = kind_prop.get("const")
    enum = kind_prop.get("enum")
    assert const == expected_kind or enum == [expected_kind], (
        f"{cls.__name__} kind literal mismatch · schema={kind_prop}"
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
        ("augment", []),  # empty options must still fall back
        ("augment", ["single"]),  # fewer than 3, should be padded
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
    advice = llm._fallback(ctx, reason=f"unit test · {kind}")

    # basic field checks
    assert advice.kind == kind
    assert advice.confidence == 0.0
    assert "降级" in advice.reasoning
    assert f"unit test · {kind}" in advice.reasoning

    # correct type
    assert isinstance(advice, ADVICE_CLASSES[kind])

    # kind-specific validity
    if isinstance(advice, AugmentAdvice):
        assert len(advice.ranked) == 3  # always pad/trunc to 3
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
        assert "S17" in text, f"{kind} prompt should explicitly declare S17"
        assert "## 输出 schema" in text
        assert "## 输出质量硬标准" in text


def test_prompt_without_knowledge_uses_fallback_text():
    ctx = DecisionContext(kind="augment", options=["法师之力", "复利", "攻速强化"])
    text = PROMPT_BUILDERS["augment"](ctx, None)
    assert "无知识库" in text


def test_prompt_with_knowledge_injects_version_context():
    ctx = DecisionContext(kind="augment", options=["法师之力", "复利", "攻速强化"])
    fake = _FakeKnowledge(ctx_text="version S17-TEST content")
    text = PROMPT_BUILDERS["augment"](ctx, fake)
    assert "version S17-TEST content" in text


def test_carousel_prompt_uses_comps_table():
    ctx = DecisionContext(kind="carousel", options=["安妮", "亚索"])
    fake = _FakeKnowledge(comps_text="| super comp | S |")
    text = PROMPT_BUILDERS["carousel"](ctx, fake)
    assert "super comp" in text


# ==================== DecisionLLM.decide · mock httpx ====================

class _MockTransport(httpx.AsyncBaseTransport):
    """An httpx transport that returns a preset response."""

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
    """Return an AsyncClient-constructing callable, used to monkeypatch httpx.AsyncClient."""
    original = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    return factory


def test_decide_happy_path(monkeypatch):
    """When the mock vLLM returns valid AugmentAdvice JSON, decide should parse it successfully."""
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

    # the request body should contain the guided_json schema and point at /chat/completions
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert "extra_body" in call["json"]
    assert "guided_json" in call["json"]["extra_body"]
    # hard constraint: max_tokens <= 500
    assert call["json"]["max_tokens"] <= 500


def test_decide_timeout_falls_back(monkeypatch):
    """When httpx raises TimeoutException, decide should return a fallback and not raise."""
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
    """When the LLM emits invalid JSON, decide should fall back and not raise."""
    ws = _blank_ws()
    ctx = DecisionContext(kind="shop", options=[])

    transport = _MockTransport({
        "choices": [
            {"message": {"content": "this is natural language, not JSON"}}
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
    """When the LLM emits valid JSON but is missing fields, pydantic validation fails and it should fall back."""
    ws = _blank_ws()
    ctx = DecisionContext(kind="item", options=[])

    transport = _MockTransport({
        "choices": [
            {"message": {"content": json.dumps({"kind": "item", "reasoning": "missing fields"})}}
        ],
    })
    monkeypatch.setattr(httpx, "AsyncClient", _patched_client_factory(transport))

    llm = DecisionLLM()
    advice = asyncio.run(llm.decide(ws, ctx))

    assert isinstance(advice, ItemAdvice)
    assert advice.confidence == 0.0
    assert "降级" in advice.reasoning


def test_decide_uses_knowledge_version_context(monkeypatch):
    """When knowledge is not None, decide should inject version_context into the system prompt."""
    ws = _rich_ws()
    ctx = DecisionContext(kind="augment", options=["A", "B", "C"])

    fake_k = _FakeKnowledge(ctx_text="injected version string XYZ")

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
    assert "injected version string XYZ" in sys_msg


# ==================== guided_json toggle ====================

def test_decide_without_guided_json(monkeypatch):
    """use_guided_json=False · the request carries no extra_body."""
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


# ==================== DecisionContext itself ====================

def test_decision_context_default_timeout():
    ctx = DecisionContext(kind="augment", options=["A", "B", "C"])
    assert ctx.timeout_s == 25.0


def test_decision_context_accepts_all_kinds():
    for kind in ("augment", "carousel", "shop", "level", "positioning", "item"):
        DecisionContext(kind=kind)  # type: ignore[arg-type]


# ==================== PositioningAdvice coerce fault tolerance ====================

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
    """Verify PositioningAdvice coerce fault tolerance for main_carry_row/col."""
    advice = PositioningAdvice(
        kind="positioning",
        reasoning="positioning test " * 10,
        confidence=0.7,
        main_carry_row=row_in,
        main_carry_col=col_in,
    )
    assert advice.main_carry_row == expected_row, f"[{desc}] row mismatch"
    assert advice.main_carry_col == expected_col, f"[{desc}] col mismatch"
    # the final value is always within valid bounds
    assert 0 <= advice.main_carry_row <= 3, f"[{desc}] row out of bounds"
    assert 0 <= advice.main_carry_col <= 6, f"[{desc}] col out of bounds"


def test_positioning_advice_coerce_does_not_affect_other_fields():
    """Coerce only affects row/col; it does not touch reasoning/confidence/bait_unit/notes."""
    advice = PositioningAdvice(
        kind="positioning",
        reasoning="对位分析：刺客可跳后排 · 建议主C放右侧角落 · 盖伦前排吸引仇恨。",
        confidence=0.85,
        main_carry_row="2",  # string -> 2
        main_carry_col="5",  # string -> 5
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
    """Simulate the LLM emitting string-form numbers; decide takes the normal path, not fallback."""
    import asyncio
    import json

    ws = _blank_ws()
    ctx = DecisionContext(kind="positioning", options=[])

    # LLM emits string-form coordinates (a common bad output)
    llm_reply = {
        "kind": "positioning",
        "reasoning": "对手有刺客 · 主C建议放后排角落 · 盖伦前排吸引火力 · 安妮后排输出更安全。",
        "confidence": 0.75,
        "main_carry_row": "3",   # string -> coerce -> 3
        "main_carry_col": "0",   # string -> coerce -> 0
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

    # key point: it should not fall back · confidence > 0
    assert isinstance(advice, PositioningAdvice)
    assert advice.confidence == pytest.approx(0.75), "should not fall back"
    assert advice.main_carry_row == 3
    assert advice.main_carry_col == 0


def test_positioning_coerce_with_mock_llm_outofrange_output(monkeypatch):
    """Simulate the LLM emitting out-of-range numbers; clamp after coerce, no fallback."""
    import asyncio
    import json

    ws = _blank_ws()
    ctx = DecisionContext(kind="positioning", options=[])

    # LLM emits out-of-range values
    llm_reply = {
        "kind": "positioning",
        "reasoning": "对手有巨魔 · 主C建议放最深后排右侧 · 拉开距离避免被冲脸 · 前排盖伦挡线。",
        "confidence": 0.68,
        "main_carry_row": 7,    # out of range -> clamp -> 3
        "main_carry_col": 10,   # out of range -> clamp -> 6
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
    assert advice.confidence == pytest.approx(0.68), "should not fall back"
    assert 0 <= advice.main_carry_row <= 3
    assert 0 <= advice.main_carry_col <= 6
    assert advice.main_carry_row == 3   # clamped
    assert advice.main_carry_col == 6   # clamped


# ==================== Zero hallucinated imports ====================

def test_decision_llm_source_has_no_openai_or_anthropic_sdk_import():
    """Hard constraint: decision_llm.py must not import the openai / anthropic SDKs."""
    import inspect

    import src.decision_llm as mod

    src_text = inspect.getsource(mod)
    # comments like "OpenAI-compatible" are allowed; a real import is forbidden
    for banned in ("import openai", "from openai", "import anthropic", "from anthropic"):
        assert banned not in src_text, f"forbidden: {banned}"
