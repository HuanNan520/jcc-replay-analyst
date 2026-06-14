# B5 · PyQt desktop overlay

**Assigned to**: Claude Opus 4.7 (`claude-opus-4-7`) · three-in-one engineering (PyQt framework + Win32 window alignment + WebSocket subscription) · plus a UI-design-sense requirement.
**Dependencies**: B4 (the WebSocket service must be running first).
**Estimated effort**: 1 day (including visual polish).
**Runtime**: **native Windows Python** (WSL can run a PyQt GUI but aligning the window to MuMu needs the Win32 API).
**Role in the new product positioning**: **the layer the player actually sees** — the product's face · the star of the portfolio demo video.

---

## Who you are

You are the Claude Opus 4.7 dispatched to `HuanNan520/jcc-replay-analyst` to execute B5.
B4 is already merged · you subscribe to its WebSocket to get the advice stream. Your task: **write a translucent floating window · overlaid above the MuMu emulator window · that shows the advice to the player**.

This is the product's visual face — the overlay you write is the first thing seen in the demo video · its visual quality directly affects the portfolio impression.

## Target visual

```
┌─────────────────────── MuMu game window ──────────────────────────┐
│                                                                     │
│   [game screen]                                  ┌────────────┐     │
│                                                  │ ★ Augment  │     │
│                                                  │ Pick #1    │     │
│                                                  │ Sorcerer Crest │ │
│                                                  │ Fit 88%    │     │
│                                                  │ Reasoning..│     │
│                                                  └────────────┘     │
│                                                  (300×180 translucent│
│                                                   card · gold border·│
│                                                   fade in/out)        │
└─────────────────────────────────────────────────────────────────────┘
```

Key features:
- **Frameless + always-on-top + translucent background**
- **Click-through** (mouse events aren't intercepted by the overlay · the player operates the game normally)
- **Follows the MuMu window as it moves** (Win32 FindWindow + GetWindowRect · poll every 500ms)
- **Gold border + Song-style title + sans-serif body** (continuing the project's visual language)
- **Fade in/out** (fade-in when advice arrives · fade-out after 8 seconds · or replaced by new advice)

---

## What to do

### 1. Add `src/overlay_ui.py`

```python
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QObject,
    pyqtSignal, QRect, QPoint, QSize,
)
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsOpacityEffect,
)

log = logging.getLogger(__name__)


# ==================== Win32 MuMu tracker ====================

try:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
except Exception:
    user32 = None


@dataclass
class WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def find_mumu_rect(title_contains: list[str] = ("MuMu", "模拟器")) -> Optional[WindowRect]:
    """Native Windows · finds the MuMu window coordinates. Returns None on WSL / Linux.

    NOTE: "模拟器" (emulator) is the actual Chinese window-title substring matched · kept as a functional matcher.
    """
    if user32 is None:
        return None

    found: list[WindowRect] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if any(kw in title for kw in title_contains):
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                found.append(WindowRect(rect.left, rect.top, rect.right, rect.bottom))
        return True

    user32.EnumWindows(_enum, 0)
    if not found:
        return None
    # take the largest by area (the main window)
    return max(found, key=lambda r: r.width * r.height)


# ==================== WebSocket client (qasync-free version) ====================

class AdviceSubscriber(QObject):
    """Runs websockets on a background thread · emits a Qt signal to the main thread."""

    advice_received = pyqtSignal(dict)   # carries the payload dict
    connection_state = pyqtSignal(str)   # "connected" / "disconnected" / "error"

    def __init__(self, ws_url: str):
        super().__init__()
        self.ws_url = ws_url
        self._thread: Optional[object] = None
        self._stop = False

    def start(self) -> None:
        import threading
        self._thread = threading.Thread(target=self._run_asyncio, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def _run_asyncio(self) -> None:
        import asyncio
        asyncio.run(self._ws_loop())

    async def _ws_loop(self) -> None:
        import asyncio
        try:
            import websockets
        except ImportError:
            log.error("websockets not installed · pip install websockets")
            self.connection_state.emit("error")
            return

        while not self._stop:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.connection_state.emit("connected")
                    log.info("WS connected · %s", self.ws_url)
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("type") in ("advice", "history"):
                            self.advice_received.emit(msg["payload"])
            except Exception as e:
                log.warning("WS disconnected · reconnecting in 5s · %s", e)
                self.connection_state.emit("disconnected")
                await asyncio.sleep(5)


# ==================== Advice Card Widget ====================
# NOTE: the labels below are shown in the overlay above a China-server game · kept as Chinese UI text.

KIND_DISPLAY = {
    "augment": ("★ 选增强", "#e6c17a"),
    "carousel": ("⚫ 轮抱", "#c9a45d"),
    "shop": ("◆ 商店", "#5a8b7a"),
    "level": ("▲ 升级决策", "#b3432e"),
    "positioning": ("◈ 摆位", "#9c7a3c"),
    "item": ("✦ 装备", "#c9a45d"),
}


class AdviceCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(320, 200)
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        self._kind_label = QLabel("")
        f1 = QFont("Songti SC", 12)
        f1.setWeight(QFont.Weight.Medium)
        self._kind_label.setFont(f1)

        self._rec_label = QLabel("")
        f2 = QFont("Songti SC", 18)
        f2.setWeight(QFont.Weight.Normal)
        self._rec_label.setFont(f2)
        self._rec_label.setWordWrap(True)

        self._reason_label = QLabel("")
        f3 = QFont("PingFang SC", 10)
        self._reason_label.setFont(f3)
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #a39d8e;")

        self._conf_label = QLabel("")
        f4 = QFont("Baskerville", 9)
        f4.setItalic(True)
        self._conf_label.setFont(f4)
        self._conf_label.setStyleSheet("color: #6b6458;")

        layout.addWidget(self._kind_label)
        layout.addWidget(self._rec_label)
        layout.addWidget(self._reason_label, 1)
        layout.addWidget(self._conf_label)

        # animations
        self._fade_in = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_in.setDuration(400)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(0.95)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_out = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_out.setDuration(600)
        self._fade_out.setStartValue(0.95)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InCubic)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out.start)

        self._accent = QColor("#e6c17a")

    def show_advice(self, payload: dict, display_ms: int = 8000) -> None:
        kind = payload.get("kind", "?")
        label, color = KIND_DISPLAY.get(kind, (f"◇ {kind}", "#c9a45d"))
        self._accent = QColor(color)
        self._kind_label.setText(label)
        self._kind_label.setStyleSheet(f"color: {color};")

        rec = payload.get("recommendation") or payload.get("action") or "—"
        self._rec_label.setText(str(rec))

        reason = payload.get("reasoning", "")
        self._reason_label.setText(reason[:160])

        conf = payload.get("confidence", 0)
        self._conf_label.setText(f"confidence · {conf:.0%}")

        self._hide_timer.stop()
        self._fade_in.start()
        self._hide_timer.start(display_ms)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # translucent dark base
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor(19, 17, 28, 235))
        bg.setColorAt(1, QColor(11, 9, 18, 235))
        p.fillRect(self.rect(), QBrush(bg))
        # gold border
        pen = QPen(self._accent)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        # top accent short line
        p.fillRect(QRect(0, 0, 60, 2), self._accent)


# ==================== Main Overlay Window ====================

class OverlayWindow(QMainWindow):
    def __init__(self, ws_url: str, click_through: bool = True):
        super().__init__()
        self._click_through = click_through
        self._setup_window()
        self._setup_ui()
        self._setup_subscriber(ws_url)
        self._setup_mumu_tracker()

    def _setup_window(self):
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if self._click_through:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _setup_ui(self):
        central = QWidget()
        central.setStyleSheet("background: transparent;")
        self.setCentralWidget(central)
        self._card = AdviceCard(central)
        # default to the top-right corner
        self._card.move(central.width() - 340, 20)

    def _setup_subscriber(self, ws_url: str):
        self._sub = AdviceSubscriber(ws_url)
        self._sub.advice_received.connect(self._on_advice)
        self._sub.connection_state.connect(self._on_state)
        self._sub.start()

    def _on_advice(self, payload: dict):
        log.info("advice received · kind=%s", payload.get("kind"))
        self._card.show_advice(payload)

    def _on_state(self, state: str):
        log.info("WS state · %s", state)

    def _setup_mumu_tracker(self):
        self._align_timer = QTimer(self)
        self._align_timer.timeout.connect(self._align_to_mumu)
        self._align_timer.start(500)
        self._align_to_mumu()

    def _align_to_mumu(self):
        rect = find_mumu_rect()
        if rect is None:
            # MuMu not found · dock to the top-right of the screen
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen.width() - 360, 40, 360, 220)
            self._card.move(20, 20)
            return
        # dock to the inside top-right of MuMu
        self.setGeometry(rect.left, rect.top, rect.width, rect.height)
        card_x = rect.width - self._card.width() - 24
        card_y = 20
        self._card.move(card_x, card_y)


# ==================== CLI ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default="ws://localhost:8765/ws/advice")
    ap.add_argument("--no-click-through", action="store_true", help="let the overlay receive the mouse (for debugging)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    win = OverlayWindow(args.ws_url, click_through=not args.no_click_through)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

### 2. Update `requirements-windows.txt`

```
PyQt6>=6.7
websockets>=12.0
```

(Not added to the main requirements.txt · because PyQt6 installs slowly under Linux headless for no benefit)

### 3. Unit test `tests/test_overlay_headless.py`

PyQt is hard to test in headless CI · so test pure functions only:

```python
import pytest


def test_find_mumu_rect_on_linux_returns_none():
    """On WSL / Linux user32 is unavailable · should return None gracefully."""
    from src.overlay_ui import find_mumu_rect, user32
    if user32 is not None:
        pytest.skip("windows only")
    assert find_mumu_rect() is None


def test_kind_display_maps_all_six_kinds():
    from src.overlay_ui import KIND_DISPLAY
    for kind in ("augment", "carousel", "shop", "level", "positioning", "item"):
        assert kind in KIND_DISPLAY
        label, color = KIND_DISPLAY[kind]
        assert label and color.startswith("#")


def test_windowrect_width_height():
    from src.overlay_ui import WindowRect
    r = WindowRect(100, 200, 900, 700)
    assert r.width == 800
    assert r.height == 500
```

This file can run on Linux in CI (user32 is None · test_find_mumu_rect_on_linux only asserts None), but the PyQt6 import should skip at import time if not installed. Use `pytest.importorskip("PyQt6", reason="Windows UI")` at the top of the file.

**Better**: at the top of `tests/test_overlay_headless.py`:
```python
import pytest
pytest.importorskip("PyQt6", reason="Windows UI only")
```

### 4. Update `.github/workflows/ci.yml`

In the pip install step, explicitly skip PyQt6 (don't let requirements drag it in indirectly):

```yaml
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pydantic httpx pillow numpy pytest fastapi uvicorn websockets
          # PyQt6 is Windows-only · not installed in CI
      - name: Run tests
        run: |
          pytest tests/ -v --ignore=tests/test_overlay_headless.py
```

Or keep the overlay test but rely on pytest.importorskip to skip it automatically · either works. Pick one.

### 5. Final README section: the full real-time coach startup flow

```markdown
### Real-time coach mode · full startup (4 terminals)

```powershell
# Terminal 1 · WSL · start vLLM
source ~/jcc-replay-analyst/.venv/bin/activate
python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-VL-4B-FP8 --port 8000

# Terminal 2 · Windows or WSL · start the advice server
python -m src.advice_server --port 8765

# Terminal 3 · native Windows Python · start OBS virtual cam + live tick (requires OBS Start Virtual Camera first)
python -m src.live_tick --fps 2 --advice-server http://localhost:8765

# Terminal 4 · native Windows Python · start the overlay
python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice
```

Play TFT · the overlay automatically floats above the MuMu window · and pops advice when a decision point triggers.
```

---

## What not to do

- Don't introduce Electron / Tauri / a web UI framework · just PyQt6
- Don't write your own WebSocket reconnect exponential backoff — a fixed 5s interval is enough
- Don't add a config file / YAML · CLI args are enough
- Don't build a "config panel inside the overlay" · v1 only displays
- Don't handle multiple MuMu instances · take the largest by area
- Don't add screenshot / recording features · irrelevant to the real-time coach
- Don't modify B4's `src/advice_server.py` — if there's a new need, tell the user, don't change the server
- Don't modify `src/schema.py` · or the other src/ files

---

## Self-acceptance checklist

Linux/WSL side (what CI can run):
- [ ] `python -c "from src.overlay_ui import find_mumu_rect, WindowRect, KIND_DISPLAY"` no error
- [ ] `pytest tests/test_overlay_headless.py -v` all green (under Linux find_mumu_rect should return None)
- [ ] `pytest tests/ -v --ignore=tests/test_overlay_headless.py` the existing 40+ tests have zero regressions

Windows side (tested on the user's machine · at minimum give them the commands to verify):
- [ ] `pip install -r requirements-windows.txt` no error
- [ ] With advice_server running · `python -m src.overlay_ui` starts · the window is transparent and the desktop is visible
- [ ] With MuMu open · the overlay window **automatically** docks to MuMu (coordinates aligned)
- [ ] Manually push with `curl -X POST http://localhost:8765/advice -d '{...one valid advice...}'` · the overlay's top-right card **fades in** · fades out after 8 seconds
- [ ] The overlay doesn't intercept the mouse (clicking on the overlay sends the mouse event to the MuMu window) — with `--no-click-through` debug mode you can click the overlay itself
- [ ] Shut down advice_server · the overlay doesn't crash · the log shows "WS disconnected · reconnecting in 5s"

`git diff --stat` only contains:
- `src/overlay_ui.py` (new)
- `tests/test_overlay_headless.py` (new)
- `requirements-windows.txt` (appended)
- `.github/workflows/ci.yml` (optional change · explicitly skip PyQt6)
- `README.md` (add a section)

## After completion

Give the user a ≤ 200-word report:
- An overlay screenshot (save to `/tmp/overlay_demo.png` or have them screenshot it manually) · at least describe the visual effect
- MuMu window alignment behavior (does it follow movement smoothly)
- What each kind's card looks like (are all 6 kinds UI-compatible)
- The user's desktop environment fit (resolution · scaling · any multi-monitor issues)
- Tuning TODOs (e.g. animation · multi-card stacking · adaptive font size)

No git commit.

---

## References

- PyQt6 Window Flags: https://doc.qt.io/qt-6/qt.html#WindowType-enum
- WA_TransparentForMouseEvents click-through: https://doc.qt.io/qt-6/qt.html#WidgetAttribute-enum
- Win32 EnumWindows + GetWindowRect: call user32.dll directly via ctypes · no pywin32 · fewer dependencies
- Font stack: continue pitch/index.html's Songti SC / Baskerville / PingFang SC combination
