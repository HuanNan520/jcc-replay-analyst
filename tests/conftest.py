"""Shared pytest fixtures for jcc-replay-analyst tests."""
import io

import pytest
from PIL import Image


@pytest.fixture
def tiny_png_bytes():
    """一张 8×8 纯色 PNG 的 bytes · 给 frame_monitor 用。"""
    img = Image.new("RGB", (8, 8), (120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def solid_frame_factory():
    """生成指定颜色的 size×size PNG bytes · 用于测 dhash 汉明距离。"""
    def _make(rgb: tuple[int, int, int], size: int = 200) -> bytes:
        img = Image.new("RGB", (size, size), rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return _make


@pytest.fixture
def stripe_frame_bytes():
    """垂直条纹图 · dhash 相邻像素差分非零 · 用来和纯色图对比出汉明距离。"""
    img = Image.new("RGB", (200, 200))
    px = img.load()
    for y in range(200):
        for x in range(200):
            v = 255 if (x // 25) % 2 == 0 else 0
            px[x, y] = (v, v, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
