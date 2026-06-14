from __future__ import annotations

import asyncio
import io
import logging
from typing import AsyncIterator, Optional

import cv2
from PIL import Image

log = logging.getLogger(__name__)


class OBSCaptureError(RuntimeError):
    pass


class OBSCapture:
    """Read frames from the OBS Virtual Camera · convert to PNG bytes for downstream consumers.

    Usage:
        cap = OBSCapture(fps=2.0)
        async for frame in cap.frames():
            ...  # frame is PNG bytes · same format as adb_client.screencap

    Windows only · WSL2 cannot access Windows camera devices.
    """

    def __init__(
        self,
        fps: float = 2.0,
        device_index: Optional[int] = None,
        device_name_hint: str = "OBS",
        expected_min_width: int = 1280,
    ):
        self.fps = fps
        self.device_name_hint = device_name_hint
        self.expected_min_width = expected_min_width
        self._device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._period = 1.0 / fps

    def _discover_device(self) -> int:
        """Enumerate Windows video devices · find the one whose name contains the hint.

        Uses pygrabber.dshow_graph.FilterGraph().get_input_devices() ·
        falls back to scanning cv2.VideoCapture(0..9) by resolution.
        """
        try:
            from pygrabber.dshow_graph import FilterGraph
            devices = FilterGraph().get_input_devices()
            log.info("Detected %d video device(s): %s", len(devices), devices)
            for i, name in enumerate(devices):
                if self.device_name_hint.lower() in name.lower():
                    log.info("Matched the OBS virtual camera · index=%d · name=%s", i, name)
                    return i
            raise OBSCaptureError(
                f"No video device containing '{self.device_name_hint}' was found · "
                f"available list: {devices} · make sure OBS Studio is running and the virtual camera is on (Start Virtual Camera button)"
            )
        except ImportError:
            log.warning("pygrabber is not installed · falling back to scanning devices")
            for i in range(10):
                c = cv2.VideoCapture(i)
                if c.isOpened():
                    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    c.release()
                    log.info("Device %d · %dx%d", i, w, h)
                    if w >= self.expected_min_width:
                        return i
            raise OBSCaptureError(
                f"Scanned 10 devices but found no camera with resolution >={self.expected_min_width} · install pygrabber or pass device_index manually"
            )

    def open(self) -> None:
        if self._device_index is None:
            self._device_index = self._discover_device()
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise OBSCaptureError(f"cv2.VideoCapture({self._device_index}) failed to open")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("OBS virtual camera opened · %dx%d · fps=%.1f", w, h, self.fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_once(self) -> bytes:
        """Read one frame · convert to PNG bytes · same format as adb_client.screencap."""
        if self._cap is None:
            self.open()
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise OBSCaptureError("cv2 read() failed · the OBS virtual camera may have been turned off")
        # BGR → RGB → PIL → PNG bytes
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()

    async def frames(self) -> AsyncIterator[bytes]:
        """Async frame generator · throttled to self.fps."""
        if self._cap is None:
            self.open()
        while True:
            start = asyncio.get_event_loop().time()
            try:
                yield self.read_once()
            except OBSCaptureError as e:
                log.warning("Frame read failed · retrying in 2s: %s", e)
                await asyncio.sleep(2)
                try:
                    self.close()
                    self.open()
                except Exception as reopen_err:
                    log.error("reopen failed: %s", reopen_err)
                    raise
                continue
            elapsed = asyncio.get_event_loop().time() - start
            sleep_time = max(0, self._period - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
