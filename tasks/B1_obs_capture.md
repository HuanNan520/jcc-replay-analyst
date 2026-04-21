# B1 · OBS 虚拟摄像头数据源

**分配给**：Claude Sonnet 4.6（`claude-sonnet-4-6`）· 中等复杂度 · 平台 API 探测 + 稳定帧流。
**依赖**：无 · 可和 B3 / B4 并行。
**预期工时**：2–3 小时。
**运行平台**：**Windows 原生 Python**（WSL2 访问不了 Windows 摄像头设备 · 这条是硬性）。
**新产品定位中的角色**：**实时 tick loop 的数据源**（替代原来的 ADB 截屏）。

---

## 你是谁

你是被派到 `HuanNan520/jcc-replay-analyst` 执行 B1 任务的 Claude Sonnet 4.6。
项目刚从"录屏复盘工具"升级到 **实时 AI 教练 + 自动复盘** 双入口。
A1-A4 已完成 · 现在进入 B 阶段 · 你负责**实时数据源**。

## 背景

新产品形态：
- 玩家在 Windows 上玩 MuMu 模拟器里的金铲铲
- OBS Studio 抓 MuMu 窗口 → 开启"虚拟摄像头"输出（Windows 视频设备层）
- **你的任务**：Python 脚本持续读这个虚拟摄像头 · 吐 PNG bytes 流给下游

**为什么不用 ADB 截屏**：ADB debugging 持续连接腾讯反外挂可能检测 · 有封号风险（已踩过坑）。OBS 虚拟摄像头是纯 Windows 系统层的视频设备 · Android 侧完全感知不到。

## 目标产物

```python
# 伪代码 · 下游怎么用
from src.capture_obs import OBSCapture

cap = OBSCapture(fps=2.0)
async for frame_bytes in cap.frames():
    # frame_bytes 是一张 PNG 编码的图
    # 和 adb_client.screencap() 返回格式一致
    ...
```

## 具体要做

### 1. 新增 `src/capture_obs.py`

```python
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
```

### 2. 新增 `requirements-windows.txt`

创建 Windows 专用依赖文件：

```
# requirements-windows.txt
# Windows 原生 Python 专用 · 实时 coach pipeline 的数据源 + UI 会用到
# 和 requirements.txt 一起 pip install · 不替换

-r requirements.txt
pygrabber>=0.2.0     ; sys_platform == "win32"
# opencv-python 已在主 requirements · 这里只加 Windows 特有的
```

### 3. CLI smoke test

新增 `scripts/test_obs_capture.py`：

```python
"""手动验证 OBS 虚拟摄像头接入。

前置：
  1. OBS Studio 已启动
  2. 在 OBS 里添加 "Window Capture" 或 "Game Capture" 抓 MuMu 窗口
  3. 按 OBS 右下角 "Start Virtual Camera" 启动虚拟摄像头

跑：
  python scripts/test_obs_capture.py --out /tmp/obs_test.png
"""
import argparse
import asyncio
import logging
from pathlib import Path

from src.capture_obs import OBSCapture


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/obs_test.png"))
    ap.add_argument("--count", type=int, default=3, help="抓几帧验证稳定性")
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with OBSCapture(fps=args.fps) as cap:
        for i in range(args.count):
            b = cap.read_once()
            path = args.out.with_name(f"{args.out.stem}_{i:02d}.png")
            path.write_bytes(b)
            print(f"✓ frame {i} · {len(b):,} bytes · saved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 4. README 更新（仅加一小段）

在 "运行时 · 默认本地推理" 段之后加：

```markdown
### 实时 coach 模式 · 数据源

实时模式要求 Windows 原生 Python（不在 WSL） · 前置：

1. 装 OBS Studio 并启动
2. 添加 "Window Capture" 抓 MuMu 窗口
3. OBS 右下 "Start Virtual Camera"
4. `pip install -r requirements-windows.txt`
5. `python scripts/test_obs_capture.py` 验证接入

更详细的实时 coach 启动流程见后续 B2/B5 任务完成后的 README 更新。
```

---

## 禁止做的事

- 不要接 Android 侧 / ADB · 这就是本任务的 raison d'être · 走回头路就是犯规
- 不要加 FFmpeg / GStreamer 等重量级依赖 · `cv2 + pygrabber` 够
- 不要改 `src/adb_client.py`（保留给后续 optional 路线）
- 不要改感知层（`vlm_client` / `ocr_client` / `frame_monitor`）· 它们吃 PNG bytes · 你的输出对齐就行
- 不要写重试无限循环 · 上面代码里 2s 重试一次是够了 · 死循环让 tick loop 去处理
- 不要引入异步回调风格 · 就用 async iterator

---

## 自验收清单

- [ ] 在 Windows 原生 Python 环境（非 WSL）执行 `python scripts/test_obs_capture.py --count 3` 
- [ ] 三张 PNG 都保存成功 · 分辨率 >= MuMu 画面实际分辨率
- [ ] PNG 能被 PIL 打开 · `Image.open(path).size` 合理（横屏 > 1200×600）
- [ ] `_discover_device` 返回的 index 对应 OBS 虚拟摄像头（debug log 打印设备列表确认）
- [ ] OBS 关闭虚拟摄像头时 `read_once` 抛 OBSCaptureError · `frames()` 会重试
- [ ] `git diff --stat` 只含：
  - `src/capture_obs.py` (新)
  - `scripts/test_obs_capture.py` (新)
  - `requirements-windows.txt` (新)
  - `README.md` (改 · 一小段)
- [ ] grep anthropic/openai 在新文件里零命中

## 完成后

给用户 ≤ 150 字报告：
- OBS Virtual Camera 在用户机器上的 device index · device name · 分辨率
- 3 张测试帧文件大小
- 有没有踩到 pygrabber 安装坑（用户之前 pip 在 fakeip 环境可能慢）
- 给 B2 的 frame source 接口契约确认（`async def frames() -> AsyncIterator[bytes]`）

不自动 git commit。

---

## 参考

- OBS Virtual Camera 官方文档: https://obsproject.com/kb/virtual-camera-guide
- pygrabber 仓库: https://github.com/andreaschiavinato/python_video_stab/tree/master/pygrabber
- cv2.VideoCapture DSHOW 后端: 用 `cv2.CAP_DSHOW` 参数 · 比默认 MSMF 更兼容虚拟摄像头
