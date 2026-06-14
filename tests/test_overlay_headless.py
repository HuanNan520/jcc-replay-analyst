"""Pure-function unit tests for overlay_ui · the whole file is skipped if PyQt6 is not installed.

PyQt UI is hard to test headless · we only verify:
  - the Win32 bridge degrades gracefully and returns None on Linux
  - KIND_DISPLAY has all six decision kinds · color/symbol non-empty
  - the width/height computation of the WindowRect dataclass

UI acceptance on the Windows side follows a manual checklist (see task spec B5).
"""
from __future__ import annotations

import pytest

# PyQt6 is primarily for Windows · if not installed on Linux/WSL, skip the whole file
pytest.importorskip("PyQt6", reason="Windows UI only · PyQt6 not installed")


def test_find_mumu_rect_on_linux_returns_none():
    """On WSL / Linux user32 is unavailable · should return None gracefully (no exception)."""
    from src.overlay_ui import find_mumu_rect, user32
    if user32 is not None:
        pytest.skip("windows only · user32 is available here")
    assert find_mumu_rect() is None


def test_kind_display_maps_all_six_kinds():
    """All six decision kinds are in KIND_DISPLAY · label/color format is correct."""
    from src.overlay_ui import KIND_DISPLAY
    expected = ("augment", "carousel", "shop", "level", "positioning", "item")
    for kind in expected:
        assert kind in KIND_DISPLAY, f"missing kind · {kind}"
        label, color = KIND_DISPLAY[kind]
        assert label, f"empty label for {kind}"
        assert color.startswith("#"), f"color must be hex · got {color}"
        assert len(color) == 7, f"color must be #RRGGBB · got {color}"


def test_windowrect_width_height():
    """WindowRect.width/height is derived correctly from left/top/right/bottom."""
    from src.overlay_ui import WindowRect
    r = WindowRect(left=100, top=200, right=900, bottom=700)
    assert r.width == 800
    assert r.height == 500


def test_windowrect_zero_size():
    """Degenerate case · zero width and zero height do not crash."""
    from src.overlay_ui import WindowRect
    r = WindowRect(left=50, top=50, right=50, bottom=50)
    assert r.width == 0
    assert r.height == 0


def test_kind_display_unknown_kind_falls_back():
    """When show_advice meets an unknown kind it should fall back to '◇ {kind}' · using the default warm gold color.

    Here we only assert the fallback-logic contract of KIND_DISPLAY.get —— actual rendering needs a QApplication,
    and we do not build a widget in headless mode.
    """
    from src.overlay_ui import KIND_DISPLAY
    assert KIND_DISPLAY.get("nonexistent_kind") is None  # triggers fallback


def test_overlay_module_imports_cleanly():
    """Top-level import of the overlay_ui module should not raise · any external dependency issue takes the fallback path."""
    import src.overlay_ui  # noqa: F401


def test_advice_subscriber_construct_without_network():
    """AdviceSubscriber construction does not connect · start() connects · construction never raises."""
    from src.overlay_ui import AdviceSubscriber
    sub = AdviceSubscriber("ws://localhost:9999/ws/nonexistent")
    assert sub.ws_url == "ws://localhost:9999/ws/nonexistent"
    assert sub._stop is False
