"""overlay_ui 的纯函数单元测试 · PyQt6 未装则整文件 skip。

PyQt UI 不好 headless 测 · 只验证：
  - Win32 桥在 Linux 下优雅降级返回 None
  - KIND_DISPLAY 六类决策齐全 · 颜色/符号非空
  - WindowRect dataclass 的 width/height 计算

Windows 侧 UI 验收走手动清单（见任务书 B5）。
"""
from __future__ import annotations

import pytest

# PyQt6 Windows 为主 · Linux/WSL 下没装就 skip 整个文件
pytest.importorskip("PyQt6", reason="Windows UI only · PyQt6 not installed")


def test_find_mumu_rect_on_linux_returns_none():
    """WSL / Linux 下 user32 不可用 · 应优雅返回 None（不抛异常）。"""
    from src.overlay_ui import find_mumu_rect, user32
    if user32 is not None:
        pytest.skip("windows only · user32 is available here")
    assert find_mumu_rect() is None


def test_kind_display_maps_all_six_kinds():
    """六类决策都在 KIND_DISPLAY · label/color 格式正确。"""
    from src.overlay_ui import KIND_DISPLAY
    expected = ("augment", "carousel", "shop", "level", "positioning", "item")
    for kind in expected:
        assert kind in KIND_DISPLAY, f"missing kind · {kind}"
        label, color = KIND_DISPLAY[kind]
        assert label, f"empty label for {kind}"
        assert color.startswith("#"), f"color must be hex · got {color}"
        assert len(color) == 7, f"color must be #RRGGBB · got {color}"


def test_windowrect_width_height():
    """WindowRect.width/height 从 left/top/right/bottom 正确推导。"""
    from src.overlay_ui import WindowRect
    r = WindowRect(left=100, top=200, right=900, bottom=700)
    assert r.width == 800
    assert r.height == 500


def test_windowrect_zero_size():
    """退化情况 · 零宽零高不崩。"""
    from src.overlay_ui import WindowRect
    r = WindowRect(left=50, top=50, right=50, bottom=50)
    assert r.width == 0
    assert r.height == 0


def test_kind_display_unknown_kind_falls_back():
    """show_advice 遇到不认识的 kind 应降级为 '◇ {kind}' · 颜色用默认暖金。

    这里只断言 KIND_DISPLAY.get 的 fallback 逻辑约定 —— 真正渲染得开 QApplication
    · 我们不在 headless 下构建 widget。
    """
    from src.overlay_ui import KIND_DISPLAY
    assert KIND_DISPLAY.get("nonexistent_kind") is None  # 触发 fallback


def test_overlay_module_imports_cleanly():
    """overlay_ui 模块 top-level 导入不应抛 · 任何外部依赖问题走降级路径。"""
    import src.overlay_ui  # noqa: F401


def test_advice_subscriber_construct_without_network():
    """AdviceSubscriber 构造不连网 · start() 才连 · 构造永不抛。"""
    from src.overlay_ui import AdviceSubscriber
    sub = AdviceSubscriber("ws://localhost:9999/ws/nonexistent")
    assert sub.ws_url == "ws://localhost:9999/ws/nonexistent"
    assert sub._stop is False
