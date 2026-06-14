"""Shared pytest fixtures for jcc-replay-analyst tests."""
import io

import pytest
from PIL import Image


@pytest.fixture
def tiny_png_bytes():
    """Bytes of an 8x8 solid-color PNG, used by frame_monitor tests."""
    img = Image.new("RGB", (8, 8), (120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def solid_frame_factory():
    """Produce size x size solid-color PNG bytes, for testing dhash Hamming distance."""
    def _make(rgb: tuple[int, int, int], size: int = 200) -> bytes:
        img = Image.new("RGB", (size, size), rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return _make


@pytest.fixture
def stripe_frame_bytes():
    """Vertical-stripe image; non-zero dhash adjacent-pixel diffs, used to measure Hamming distance against a solid image."""
    img = Image.new("RGB", (200, 200))
    px = img.load()
    for y in range(200):
        for x in range(200):
            v = 255 if (x // 25) % 2 == 0 else 0
            px[x, y] = (v, v, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
