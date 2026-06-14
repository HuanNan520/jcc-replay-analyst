"""Unit tests · src/capture_obs.py · mock verification inside WSL.

All cv2.VideoCapture calls are mocked out · no dependency on a real camera device.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.capture_obs import OBSCapture, OBSCaptureError


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_fake_bgr(w: int = 1920, h: int = 1080) -> np.ndarray:
    """Produce an all-blue BGR image · used to mock cap.read()."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # B channel
    return frame


def _make_mock_cap(w: int = 1920, h: int = 1080, read_ok: bool = True):
    """Return a mock cv2.VideoCapture object with controllable behavior."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: float(w),   # CAP_PROP_FRAME_WIDTH  = 3
        4: float(h),   # CAP_PROP_FRAME_HEIGHT = 4
    }.get(prop, 0.0)
    cap.read.return_value = (read_ok, _make_fake_bgr(w, h) if read_ok else None)
    return cap


# ── open / close tests ────────────────────────────────────────────────────────


def test_open_with_explicit_device_index():
    """When device_index is passed, use it directly · skip device discovery."""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()) as mock_vc:
        cap = OBSCapture(fps=2.0, device_index=1)
        cap.open()
        mock_vc.assert_called_once()
        # the first positional argument is device_index=1
        assert mock_vc.call_args[0][0] == 1
        cap.close()


def test_open_raises_when_cap_not_opened():
    """When VideoCapture.isOpened() is False, open() raises OBSCaptureError."""
    bad_cap = MagicMock()
    bad_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=bad_cap):
        cap = OBSCapture(device_index=0)
        with pytest.raises(OBSCaptureError, match="打开失败"):
            cap.open()


def test_close_idempotent():
    """Repeated close() does not raise."""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        cap = OBSCapture(device_index=0)
        cap.open()
        cap.close()
        cap.close()  # the second close should be a no-op


# ── read_once tests ───────────────────────────────────────────────────────────


def test_read_once_returns_valid_png():
    """The bytes returned by read_once() are a valid PNG · PIL can open it · size matches."""
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
    """When cv2.read() returns ok=False, read_once() raises OBSCaptureError."""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap(read_ok=False)):
        cap = OBSCapture(device_index=0)
        cap.open()
        with pytest.raises(OBSCaptureError, match="read\\(\\) 失败"):
            cap.read_once()
        cap.close()


def test_read_once_auto_opens():
    """Calling read_once() without open() · should auto-trigger open()."""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        cap = OBSCapture(device_index=0)
        # do not open · read directly
        png = cap.read_once()
        assert len(png) > 0
        cap.close()


# ── context manager tests ─────────────────────────────────────────────────────


def test_context_manager():
    """with OBSCapture() as cap: should auto open / close."""
    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        with OBSCapture(device_index=0) as cap:
            png = cap.read_once()
            assert len(png) > 0
        # after exit, _cap should be None
        assert cap._cap is None


# ── _discover_device tests ────────────────────────────────────────────────────


def test_discover_device_via_pygrabber():
    """When pygrabber is available · should return the device index containing the 'OBS' hint."""
    mock_fg = MagicMock()
    mock_fg.get_input_devices.return_value = ["Webcam HD", "OBS Virtual Camera", "Other Device"]

    with patch.dict("sys.modules", {"pygrabber": MagicMock(), "pygrabber.dshow_graph": MagicMock()}):
        with patch("pygrabber.dshow_graph.FilterGraph", return_value=mock_fg):
            cap = OBSCapture(device_name_hint="OBS")
            idx = cap._discover_device()
            assert idx == 1  # "OBS Virtual Camera" is at index 1


def test_discover_device_hint_not_found_raises():
    """When pygrabber is available but no device matches the hint · raises OBSCaptureError."""
    mock_fg = MagicMock()
    mock_fg.get_input_devices.return_value = ["Webcam HD", "Other Cam"]

    with patch.dict("sys.modules", {"pygrabber": MagicMock(), "pygrabber.dshow_graph": MagicMock()}):
        with patch("pygrabber.dshow_graph.FilterGraph", return_value=mock_fg):
            cap = OBSCapture(device_name_hint="OBS")
            with pytest.raises(OBSCaptureError, match="未找到包含"):
                cap._discover_device()


def test_discover_device_fallback_by_resolution():
    """When pygrabber is unavailable · fall back to iterating · find a device with width >= expected_min_width."""
    # device 0: 640x480 · device 1: 1920x1080
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
        # ImportError -> pygrabber-unavailable path
        with patch("builtins.__import__", side_effect=ImportError("no pygrabber")):
            pass  # test the fallback directly

        # call _discover_device directly and make the pygrabber import raise ImportError
        cap = OBSCapture(device_name_hint="OBS", expected_min_width=1280)

        with patch.dict("sys.modules", {"pygrabber": None, "pygrabber.dshow_graph": None}):
            # sys.modules[key]=None makes the import raise ImportError
            idx = cap._discover_device()
            assert idx == 1


def test_discover_device_fallback_no_match_raises():
    """When the fallback iterates 10 devices and none meet the resolution requirement · raises OBSCaptureError."""
    mock_cap_small = MagicMock()
    mock_cap_small.isOpened.return_value = True
    mock_cap_small.get.side_effect = lambda p: {3: 640.0, 4: 480.0}.get(p, 0.0)

    with patch("cv2.VideoCapture", return_value=mock_cap_small):
        cap = OBSCapture(device_name_hint="OBS", expected_min_width=1280)
        with patch.dict("sys.modules", {"pygrabber": None, "pygrabber.dshow_graph": None}):
            with pytest.raises(OBSCaptureError, match="遍历 10 个设备"):
                cap._discover_device()


# ── frames() async-generator tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_frames_yields_png():
    """frames() can yield at least one valid PNG frame."""
    import asyncio

    with patch("cv2.VideoCapture", return_value=_make_mock_cap()):
        cap = OBSCapture(device_index=0, fps=30.0)  # high fps · sleep approaches 0
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
    """After a read() failure, frames() closes/reopens and retries."""
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
        # the first next fails · sleep 2s · reopen -> ok_cap · then another frame
        # to avoid a real 2s sleep · patch asyncio.sleep
        with patch("asyncio.sleep", return_value=None):
            frame = await frames_gen.__anext__()

        img = Image.open(io.BytesIO(frame))
        assert img.format == "PNG"
        cap.close()
