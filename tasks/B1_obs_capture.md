# B1 · OBS Virtual Camera data source

**Assigned to**: Claude Sonnet 4.6 (`claude-sonnet-4-6`) · medium complexity · platform-API probing + stable frame stream.
**Dependencies**: none · can run in parallel with B3 / B4.
**Estimated effort**: 2–3 hours.
**Runtime platform**: **native Windows Python** (WSL2 cannot access Windows camera devices · this is a hard requirement).
**Role in the new product positioning**: **the data source for the real-time tick loop** (replaces the old ADB screencap).

---

## Who you are

You are the Claude Sonnet 4.6 dispatched to `HuanNan520/jcc-replay-analyst` to execute task B1.
The project just upgraded from a "screen-recording replay tool" into a dual-entry **real-time AI coach + automatic replay**.
A1-A4 are done · now we enter phase B · you own the **real-time data source**.

## Background

The new product form:
- The player plays TFT inside the MuMu emulator on Windows
- OBS Studio captures the MuMu window → enables "Virtual Camera" output (the Windows video-device layer)
- **Your task**: a Python script that continuously reads this virtual camera · emitting a stream of PNG bytes to downstream

**Why not ADB screencap**: a persistent ADB debugging connection may be detected by Tencent anti-cheat · ban risk (already been burned). The OBS Virtual Camera is a pure Windows-system-layer video device · the Android side cannot perceive it at all.

## Target deliverable

```python
# pseudocode · how downstream uses it
from src.capture_obs import OBSCapture

cap = OBSCapture(fps=2.0)
async for frame_bytes in cap.frames():
    # frame_bytes is one PNG-encoded image
    # same format as adb_client.screencap() returns
    ...
```

## What to do

### 1. Add `src/capture_obs.py`

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
    """Reads frames from the OBS Virtual Camera · converts them to PNG bytes for downstream.

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
        """Enumerates Windows video devices · finds the one whose name contains the hint.

        Uses pygrabber.dshow_graph.FilterGraph().get_input_devices() ·
        falls back to iterating cv2.VideoCapture(0..9) and checking resolution.
        """
        try:
            from pygrabber.dshow_graph import FilterGraph
            devices = FilterGraph().get_input_devices()
            log.info("detected %d video devices: %s", len(devices), devices)
            for i, name in enumerate(devices):
                if self.device_name_hint.lower() in name.lower():
                    log.info("matched OBS Virtual Camera · index=%d · name=%s", i, name)
                    return i
            raise OBSCaptureError(
                f"no video device containing '{self.device_name_hint}' found · "
                f"available list: {devices} · make sure OBS Studio is running with the virtual camera on (Start Virtual Camera button)"
            )
        except ImportError:
            log.warning("pygrabber not installed · falling back to device iteration")
            for i in range(10):
                c = cv2.VideoCapture(i)
                if c.isOpened():
                    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    c.release()
                    log.info("device %d · %dx%d", i, w, h)
                    if w >= self.expected_min_width:
                        return i
            raise OBSCaptureError(
                f"iterated 10 devices, none with resolution >={self.expected_min_width} · install pygrabber or pass device_index manually"
            )

    def open(self) -> None:
        if self._device_index is None:
            self._device_index = self._discover_device()
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise OBSCaptureError(f"cv2.VideoCapture({self._device_index}) failed to open")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("OBS Virtual Camera opened · %dx%d · fps=%.1f", w, h, self.fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_once(self) -> bytes:
        """Reads one frame · converts to PNG bytes · same format as adb_client.screencap."""
        if self._cap is None:
            self.open()
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise OBSCaptureError("cv2 read() failed · the OBS Virtual Camera may have been turned off")
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
                log.warning("frame read failed · retrying in 2s: %s", e)
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
```

### 2. Add `requirements-windows.txt`

Create a Windows-specific dependency file:

```
# requirements-windows.txt
# Native Windows Python only · used by the real-time coach pipeline's data source + UI
# pip install alongside requirements.txt · does not replace it

-r requirements.txt
pygrabber>=0.2.0     ; sys_platform == "win32"
# opencv-python is already in the main requirements · this only adds the Windows-specific bits
```

### 3. CLI smoke test

Add `scripts/test_obs_capture.py`:

```python
"""Manually verify the OBS Virtual Camera connection.

Prerequisites:
  1. OBS Studio is running
  2. Add a "Window Capture" or "Game Capture" in OBS to grab the MuMu window
  3. Click "Start Virtual Camera" at the bottom right of OBS

Run:
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
    ap.add_argument("--count", type=int, default=3, help="how many frames to grab to check stability")
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

### 4. README update (add a small section only)

After the "Runtime · local inference by default" section, add:

```markdown
### Real-time coach mode · data source

Real-time mode requires native Windows Python (not WSL) · prerequisites:

1. Install OBS Studio and launch it
2. Add a "Window Capture" to grab the MuMu window
3. Click "Start Virtual Camera" at the bottom right of OBS
4. `pip install -r requirements-windows.txt`
5. `python scripts/test_obs_capture.py` to verify the connection

The fuller real-time coach startup flow is in the README update after the B2/B5 tasks are done.
```

---

## What not to do

- Do not touch the Android side / ADB · this is the task's raison d'être · backtracking is a foul
- Do not add heavyweight deps like FFmpeg / GStreamer · `cv2 + pygrabber` is enough
- Do not modify `src/adb_client.py` (kept for a later optional route)
- Do not modify the perception layer (`vlm_client` / `ocr_client` / `frame_monitor`) · they consume PNG bytes · just align your output
- Do not write an infinite retry loop · the 2s single retry above is enough · let the tick loop handle dead loops
- Do not introduce an async-callback style · just use the async iterator

---

## Self-acceptance checklist

- [ ] In a native Windows Python environment (not WSL), run `python scripts/test_obs_capture.py --count 3`
- [ ] All three PNGs save successfully · resolution >= the actual MuMu display resolution
- [ ] PNGs open in PIL · `Image.open(path).size` is reasonable (landscape > 1200×600)
- [ ] The index returned by `_discover_device` corresponds to the OBS Virtual Camera (confirm via the debug-log device list)
- [ ] When OBS turns off the virtual camera, `read_once` raises OBSCaptureError · `frames()` retries
- [ ] `git diff --stat` only contains:
  - `src/capture_obs.py` (new)
  - `scripts/test_obs_capture.py` (new)
  - `requirements-windows.txt` (new)
  - `README.md` (changed · one small section)
- [ ] grep for anthropic/openai in the new files returns zero hits

## After completion

Give the user a ≤ 150-word report:
- The OBS Virtual Camera's device index · device name · resolution on the user's machine
- File sizes of the 3 test frames
- Whether you hit any pygrabber install snags (the user's earlier pip in a fakeip environment may be slow)
- Confirm the frame-source interface contract for B2 (`async def frames() -> AsyncIterator[bytes]`)

Do not git commit automatically.

---

## References

- OBS Virtual Camera official docs: https://obsproject.com/kb/virtual-camera-guide
- pygrabber repo: https://github.com/andreaschiavinato/python_video_stab/tree/master/pygrabber
- cv2.VideoCapture DSHOW backend: pass `cv2.CAP_DSHOW` · more compatible with virtual cameras than the default MSMF
