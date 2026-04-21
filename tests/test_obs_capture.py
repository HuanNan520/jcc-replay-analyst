"""单元测试 · src/capture_obs.py · WSL 内 mock 验证。

所有 cv2.VideoCapture 调用都被 mock 掉 · 不依赖真实摄像头设备。
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.capture_obs import OBSCapture, OBSCaptureError


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _make_fake_bgr(w: int = 1920, h: int = 1080) -> np.ndarray:
    """生成一张全蓝 BGR 图 · 用于 mock cap.read()。"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # B channel
    return frame


def _make_mock_cap(w: int = 1920, h: int = 1080, read_ok: bool = True):
    """返回一个行为可控的 mock cv2.VideoCapture 对象。"""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: float(w),   # CAP_PROP_FRAME_WIDTH  = 3
        4: float(h),   # CAP_PROP_FRAME_HEIGHT = 4
    }.get(prop, 0.0)
    cap.read.return_value = (read_ok, _make_fake_bgr(w, h) if read_ok else None)
    return cap


# ── 测试 open / close ────────────────────────────────────────────────────────


def test_open_with_explicit_device_index():
    """传了 device_index 时直接用 · 不走设备发现。"""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()) as mock_vc:
        cap = OBSCapture(fps=2.0, device_index=1)
        cap.open()
        mock_vc.assert_called_once()
        # 第一个位置参数是 device_index=1
        assert mock_vc.call_args[0][0] == 1
        cap.close()


def test_open_raises_when_cap_not_opened():
    """VideoCapture.isOpened() 为 False 时 open() 抛 OBSCaptureError。"""
    bad_cap = MagicMock()
    bad_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=bad_cap):
        cap = OBSCapture(device_index=0)
        with pytest.raises(OBSCaptureError, match="打开失败"):
            cap.open()


def test_close_idempotent():
    """连续 close() 不抛异常。"""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        cap = OBSCapture(device_index=0)
        cap.open()
        cap.close()
        cap.close()  # 第二次 close 应是 no-op


# ── 测试 read_once ───────────────────────────────────────────────────────────


def test_read_once_returns_valid_png():
    """read_once() 返回的 bytes 是合法 PNG · PIL 能打开 · 尺寸匹配。"""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap(1920, 1080)):
        cap = OBSCapture(device_index=0)
        cap.open()
        png = cap.read_once()
        cap.close()

    assert isinstance(png, bytes)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.size == (1920, 1080)


def test_read_once_raises_on_failed_read():
    """cv2.read() 返回 ok=False 时 read_once() 抛 OBSCaptureError。"""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap(read_ok=False)):
        cap = OBSCapture(device_index=0)
        cap.open()
        with pytest.raises(OBSCaptureError, match="read\\(\\) 失败"):
            cap.read_once()
        cap.close()


def test_read_once_auto_opens():
    """不调 open() 直接 read_once() · 应自动触发 open()。"""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        cap = OBSCapture(device_index=0)
        # 不 open · 直接 read
        png = cap.read_once()
        assert len(png) > 0
        cap.close()


# ── 测试 context manager ─────────────────────────────────────────────────────


def test_context_manager():
    """with OBSCapture() as cap: 应自动 open / close。"""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        with OBSCapture(device_index=0) as cap:
            png = cap.read_once()
            assert len(png) > 0
        # exit 后 _cap 应为 None
        assert cap._cap is None


# ── 测试 _discover_device ────────────────────────────────────────────────────


def test_discover_device_via_pygrabber():
    """pygrabber 可用时 · 应返回含 'OBS' hint 的设备 index。"""
    mock_fg = MagicMock()
    mock_fg.get_input_devices.return_value = ["Webcam HD", "OBS Virtual Camera", "Other Device"]

    with patch.dict("sys.modules", {"pygrabber": MagicMock(), "pygrabber.dshow_graph": MagicMock()}):
        with patch("pygrabber.dshow_graph.FilterGraph", return_value=mock_fg):
            cap = OBSCapture(device_name_hint="OBS")
            idx = cap._discover_device()
            assert idx == 1  # "OBS Virtual Camera" 在 index 1


def test_discover_device_hint_not_found_raises():
    """pygrabber 可用但没有匹配 hint 的设备时 · 抛 OBSCaptureError。"""
    mock_fg = MagicMock()
    mock_fg.get_input_devices.return_value = ["Webcam HD", "Other Cam"]

    with patch.dict("sys.modules", {"pygrabber": MagicMock(), "pygrabber.dshow_graph": MagicMock()}):
        with patch("pygrabber.dshow_graph.FilterGraph", return_value=mock_fg):
            cap = OBSCapture(device_name_hint="OBS")
            with pytest.raises(OBSCaptureError, match="未找到包含"):
                cap._discover_device()


def test_discover_device_fallback_by_resolution():
    """pygrabber 不可用时 · 降级遍历 · 找到宽度 >= expected_min_width 的设备。"""
    # 设备 0: 640x480 · 设备 1: 1920x1080
    mock_cap_small = MagicMock()
    mock_cap_small.isOpened.return_value = True
    mock_cap_small.get.side_effect = lambda p: {3: 640.0, 4: 480.0}.get(p, 0.0)

    mock_cap_large = MagicMock()
    mock_cap_large.isOpened.return_value = True
    mock_cap_large.get.side_effect = lambda p: {3: 1920.0, 4: 1080.0}.get(p, 0.0)

    mock_cap_closed = MagicMock()
    mock_cap_closed.isOpened.return_value = False

    def _vc_factory(i, *args, **kwargs):
        return {0: mock_cap_small, 1: mock_cap_large}.get(i, mock_cap_closed)

    with patch("cv2.VideoCapture", side_effect=_vc_factory):
        # ImportError → pygrabber 不可用路径
        with patch("builtins.__import__", side_effect=ImportError("no pygrabber")):
            pass  # 直接测 fallback

        # 直接调 _discover_device 并让 pygrabber import 抛 ImportError
        cap = OBSCapture(device_name_hint="OBS", expected_min_width=1280)

        with patch.dict("sys.modules", {"pygrabber": None, "pygrabber.dshow_graph": None}):
            # sys.modules[key]=None 会让 import 抛 ImportError
            idx = cap._discover_device()
            assert idx == 1


def test_discover_device_fallback_no_match_raises():
    """降级遍历 10 个设备都不满足分辨率要求时 · 抛 OBSCaptureError。"""
    mock_cap_small = MagicMock()
    mock_cap_small.isOpened.return_value = True
    mock_cap_small.get.side_effect = lambda p: {3: 640.0, 4: 480.0}.get(p, 0.0)

    with patch("cv2.VideoCapture", return_value=mock_cap_small):
        cap = OBSCapture(device_name_hint="OBS", expected_min_width=1280)
        with patch.dict("sys.modules", {"pygrabber": None, "pygrabber.dshow_graph": None}):
            with pytest.raises(OBSCaptureError, match="遍历 10 个设备"):
                cap._discover_device()


# ── 测试 frames() 异步生成器 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_frames_yields_png():
    """frames() 能 yield 至少一帧合法 PNG。"""
    import asyncio

    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        cap = OBSCapture(device_index=0, fps=30.0)  # 高 fps · sleep 趋近 0
        cap.open()
        count = 0
        async for frame in cap.frames():
            img = Image.open(io.BytesIO(frame))
            assert img.format == "PNG"
            count += 1
            if count >= 2:
                break
        cap.close()
        assert count == 2


@pytest.mark.asyncio
async def test_frames_retries_on_read_failure():
    """read() 失败后 frames() 会 close/reopen 并重试。"""
    fail_cap = _make_mock_cap(read_ok=False)
    ok_cap = _make_mock_cap(read_ok=True)

    call_count = 0

    def _vc_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return fail_cap if call_count == 1 else ok_cap

    with patch("cv2.VideoCapture", side_effect=_vc_side_effect):
        cap = OBSCapture(device_index=0, fps=30.0)
        cap.open()  # call 1 → fail_cap

        frames_gen = cap.frames()
        # 第一次 next 会失败 · sleep 2s · reopen → ok_cap · 再来一帧
        # 为避免真 sleep 2s · patch asyncio.sleep
        with patch("asyncio.sleep", return_value=None):
            frame = await frames_gen.__anext__()

        img = Image.open(io.BytesIO(frame))
        assert img.format == "PNG"
        cap.close()
