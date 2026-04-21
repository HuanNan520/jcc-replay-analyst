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
    """从 OBS Virtual Camera 读帧 · 转成 PNG bytes 吐给下游。

    用法：
        cap = OBSCapture(fps=2.0)
        async for frame in cap.frames():
            ...  # frame 是 PNG bytes · 和 adb_client.screencap 同格式

    Windows only · WSL2 访问不到 Windows 摄像头设备。
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
        """枚举 Windows 视频设备 · 找名字含 hint 的那个。

        用 pygrabber.dshow_graph.FilterGraph().get_input_devices() ·
        fallback 到遍历 cv2.VideoCapture(0..9) 看分辨率。
        """
        try:
            from pygrabber.dshow_graph import FilterGraph
            devices = FilterGraph().get_input_devices()
            log.info("检测到 %d 个视频设备: %s", len(devices), devices)
            for i, name in enumerate(devices):
                if self.device_name_hint.lower() in name.lower():
                    log.info("匹配 OBS 虚拟摄像头 · index=%d · name=%s", i, name)
                    return i
            raise OBSCaptureError(
                f"未找到包含 '{self.device_name_hint}' 的视频设备 · "
                f"可选列表: {devices} · 请确认 OBS Studio 已启动且开启虚拟摄像头（Start Virtual Camera 按钮）"
            )
        except ImportError:
            log.warning("pygrabber 未安装 · 降级遍历设备")
            for i in range(10):
                c = cv2.VideoCapture(i)
                if c.isOpened():
                    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    c.release()
                    log.info("设备 %d · %dx%d", i, w, h)
                    if w >= self.expected_min_width:
                        return i
            raise OBSCaptureError(
                f"遍历 10 个设备未找到分辨率 >={self.expected_min_width} 的摄像头 · 请装 pygrabber 或手动传 device_index"
            )

    def open(self) -> None:
        if self._device_index is None:
            self._device_index = self._discover_device()
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise OBSCaptureError(f"cv2.VideoCapture({self._device_index}) 打开失败")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("OBS 虚拟摄像头已打开 · %dx%d · fps=%.1f", w, h, self.fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_once(self) -> bytes:
        """读一帧 · 转 PNG bytes · 和 adb_client.screencap 同格式。"""
        if self._cap is None:
            self.open()
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise OBSCaptureError("cv2 read() 失败 · OBS 虚拟摄像头可能被关闭")
        # BGR → RGB → PIL → PNG bytes
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()

    async def frames(self) -> AsyncIterator[bytes]:
        """异步帧生成器 · 按 self.fps 节流。"""
        if self._cap is None:
            self.open()
        while True:
            start = asyncio.get_event_loop().time()
            try:
                yield self.read_once()
            except OBSCaptureError as e:
                log.warning("帧读取失败 · 2s 后重试: %s", e)
                await asyncio.sleep(2)
                try:
                    self.close()
                    self.open()
                except Exception as reopen_err:
                    log.error("reopen 失败: %s", reopen_err)
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
