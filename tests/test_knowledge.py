"""S16 知识库 smoke test · 验证 jcc-daida 包装层基本契约。"""
import pytest

from src.knowledge import Comp, S16Knowledge, load_s16_knowledge


def test_load_or_gracefully_none():
    k = load_s16_knowledge()
    if k is None:
        pytest.skip("jcc-daida 未在环境中 · 跳过(预期行为 · 降级路径另测)")
    assert len(k.comps) >= 5
    assert len(k.all_units) >= 20
    assert len(k.all_traits) >= 10
    assert len(k.all_items) >= 20
    assert len(k.all_augments) >= 20
    first_unit = k.comps[0].core_units[0]
    assert k.validate_unit_name(first_unit)


def test_version_context_nonempty():
    k = load_s16_knowledge()
    if k is None:
        pytest.skip("jcc-daida 未安装")
    ctx = k.version_context()
    assert len(ctx) > 100
    assert "S16" in ctx or "英雄联盟" in ctx


def test_comps_table_is_markdown():
    k = load_s16_knowledge()
    if k is None:
        pytest.skip("jcc-daida 未安装")
    table = k.comps_table()
    assert "|" in table
    assert "---" in table
    assert table.count("\n") >= len(k.comps) + 1


def test_validate_unit_name_rejects_unknown():
    k = load_s16_knowledge()
    if k is None:
        pytest.skip("jcc-daida 未安装")
    assert not k.validate_unit_name("不存在的英雄名字2077")
    assert not k.validate_unit_name("")


def test_graceful_degradation_when_path_missing(monkeypatch):
    """jcc-daida 不可达时 · load_s16_knowledge 返回 None · 不抛异常。"""
    monkeypatch.setenv("JCC_DAIDA_PATH", "/nonexistent/path/xyz")
    from src import knowledge as knowledge_mod
    monkeypatch.setattr(
        knowledge_mod,
        "_DEFAULT_DAIDA_PATH",
        type(knowledge_mod._DEFAULT_DAIDA_PATH)("/also/nonexistent"),
    )
    result = knowledge_mod.load_s16_knowledge()
    assert result is None


def test_knowledge_satisfies_protocol_shape():
    """duck typing 检查 · S16Knowledge 有 KnowledgeProvider 需要的三个方法。"""
    k = load_s16_knowledge()
    if k is None:
        pytest.skip("jcc-daida 未安装")
    assert callable(getattr(k, "version_context", None))
    assert callable(getattr(k, "comps_table", None))
    assert callable(getattr(k, "validate_unit_name", None))


def test_comp_dataclass_shape():
    k = load_s16_knowledge()
    if k is None:
        pytest.skip("jcc-daida 未安装")
    c = k.comps[0]
    assert isinstance(c, Comp)
    assert c.name
    assert c.tier in {"S", "A", "B", "C"}
    assert isinstance(c.core_units, list) and len(c.core_units) > 0
    assert isinstance(c.core_items, dict)
