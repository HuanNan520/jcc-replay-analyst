"""PyQt desktop overlay · a translucent card floating over the MuMu emulator window.

The product face — the layer the player sees · subscribes to the advice server's WebSocket advice stream.

Key features:
- Frameless + always-on-top + translucent background · WA_TranslucentBackground
- Click-through (WA_TransparentForMouseEvents) · the player keeps operating the game normally
- Follows the MuMu window as it moves (Win32 FindWindow + GetWindowRect · poll 500ms)
- Gold border + Songti title + sans-serif body · continues the pitch/index.html visual language
- Fade in/out (400ms OutCubic / 600ms InCubic) · new advice replaces old · auto-fades after 8s
- WebSocket reconnect at a fixed 5s interval · does not crash when the server disconnects

Runtime (native Windows Python):
    pip install -r requirements-windows.txt
    python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice

On WSL / Linux, PyQt6 installs fine but MuMu window alignment relies on Win32 · find_mumu_rect
returns None -> the overlay degrades to the top-right corner of the screen (handy for dev/debug without MuMu).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QObject,
    pyqtSignal, QRect,
)
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QGraphicsOpacityEffect,
)

log = logging.getLogger(__name__)


# ==================== Win32 MuMu tracker ====================

try:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
except (OSError, AttributeError, ImportError):
    # WSL / Linux · no user32.dll · degrade
    user32 = None


@dataclass
class WindowRect:
    """A Win32 window rectangle · a dataclass for easy testing."""
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


def find_mumu_rect(
    # "模拟器" (= "emulator") is matched against the real Windows window title · kept Chinese.
    title_contains: tuple[str, ...] = ("MuMu", "模拟器"),
) -> Optional[WindowRect]:
    """Windows-native · find the MuMu window coordinates.

    On WSL / Linux user32 is None · returns None -> the caller degrades to the top-right of the screen.
    With multiple instances, takes the one with the largest area (the main window) · does not handle multi-boxing (v1 convention).
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
    return max(found, key=lambda r: r.width * r.height)


# ==================== WebSocket subscriber ====================

class AdviceSubscriber(QObject):
    """Runs websockets on a background thread · emits a Qt signal to the main thread.

    Avoids qasync (one fewer dependency) · spins up a daemon thread running its own asyncio.
    Reconnects at a fixed 5s interval on disconnect · no exponential backoff (server is on the same machine · simple is enough).
    """

    advice_received = pyqtSignal(dict)   # payload dict · from a history or advice message
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
            log.error("websockets is not installed · pip install websockets")
            self.connection_state.emit("error")
            return

        while not self._stop:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.connection_state.emit("connected")
                    log.info("WS connected · %s", self.ws_url)
                    async for raw in ws:
                        if raw == "pong":
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            log.debug("WS received non-JSON · ignoring · %r", raw[:60])
                            continue
                        if msg.get("type") in ("advice", "history"):
                            self.advice_received.emit(msg["payload"])
            except Exception as e:
                log.warning("WS disconnected · reconnecting in 5s · %s", e)
                self.connection_state.emit("disconnected")
                await asyncio.sleep(5)


# ==================== Advice Card Widget ====================

# Six decision types · each with its own color + symbol · continues the pitch/index.html visual.
# The Chinese labels are overlay text shown over the Chinese game, alongside the LLM's Chinese
# advice body · kept Chinese for product-UI coherence.
KIND_DISPLAY = {
    "augment":     ("★ 选增强",    "#e6c17a"),  # gold
    "carousel":    ("⚫ 轮抱",      "#c9a45d"),  # warm gold
    "shop":        ("◆ 商店",      "#5a8b7a"),  # celadon
    "level":       ("▲ 升级决策",  "#b3432e"),  # vermilion
    "positioning": ("◈ 摆位",      "#9c7a3c"),  # deep gold
    "item":        ("✦ 装备",      "#c9a45d"),  # warm gold
}


class AdviceCard(QWidget):
    """A single translucent card · 300x180 · gold border · fade in/out animation."""

    CARD_W = 320
    CARD_H = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.CARD_W, self.CARD_H)

        # Opacity effect · used for fade in/out
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        # kind label · Songti, medium size
        self._kind_label = QLabel("")
        f1 = QFont("Songti SC", 12)
        f1.setWeight(QFont.Weight.Medium)
        self._kind_label.setFont(f1)

        # main recommendation text · Songti, large size
        self._rec_label = QLabel("")
        f2 = QFont("Songti SC", 18)
        f2.setWeight(QFont.Weight.Normal)
        self._rec_label.setFont(f2)
        self._rec_label.setWordWrap(True)
        self._rec_label.setStyleSheet("color: #f0e4c8;")

        # reasoning text · sans-serif, small size
        self._reason_label = QLabel("")
        f3 = QFont("PingFang SC", 10)
        self._reason_label.setFont(f3)
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #a39d8e;")

        # confidence · Baskerville italic
        self._conf_label = QLabel("")
        f4 = QFont("Baskerville", 9)
        f4.setItalic(True)
        self._conf_label.setFont(f4)
        self._conf_label.setStyleSheet("color: #6b6458;")

        layout.addWidget(self._kind_label)
        layout.addWidget(self._rec_label)
        layout.addWidget(self._reason_label, 1)
        layout.addWidget(self._conf_label)

        # Animations
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

        # 8-second auto fade-out timer
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out.start)

        self._accent = QColor("#e6c17a")

    def show_advice(self, payload: dict, display_ms: int = 8000) -> None:
        """Received new advice · fill the card · start the fade-in · schedule the 8s fade-out.

        Field handling:
          - kind required · falls back to "◇ {kind}" when not in KIND_DISPLAY
          - recommendation / action, either one · shows "—" when neither is present
          - reasoning optional · truncated to 160 chars (the card is small)
          - confidence optional · defaults to 0
        """
        kind = payload.get("kind", "?")
        label, color = KIND_DISPLAY.get(kind, (f"◇ {kind}", "#c9a45d"))
        self._accent = QColor(color)
        self._kind_label.setText(label)
        self._kind_label.setStyleSheet(f"color: {color};")

        rec = payload.get("recommendation") or payload.get("action") or "—"
        self._rec_label.setText(str(rec))

        reason = payload.get("reasoning", "") or ""
        if len(reason) > 160:
            reason = reason[:157] + "…"
        self._reason_label.setText(reason)

        conf = payload.get("confidence", 0)
        try:
            self._conf_label.setText(f"confidence · {float(conf):.0%}")
        except (TypeError, ValueError):
            self._conf_label.setText("confidence · —")

        # Re-trigger the animation · replaces the previous item directly if it is still showing
        self._hide_timer.stop()
        self._fade_out.stop()
        self._fade_in.stop()
        self._fade_in.start()
        self._hide_timer.start(display_ms)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Translucent dark gradient background
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor(19, 17, 28, 235))
        bg.setColorAt(1, QColor(11, 9, 18, 235))
        p.fillRect(self.rect(), QBrush(bg))

        # Gold border (accent color)
        pen = QPen(self._accent)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

        # Top accent line · 2px tall, 60px wide
        p.fillRect(QRect(0, 0, 60, 2), self._accent)


# ==================== Main Overlay Window ====================

class OverlayWindow(QMainWindow):
    """Main overlay window · frameless + always-on-top + transparent · follows MuMu coordinates."""

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
            | Qt.WindowType.Tool  # Tool: stays off the taskbar
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if self._click_through:
            # Click-through · mouse events go to the window below the overlay (MuMu)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _setup_ui(self):
        central = QWidget()
        central.setStyleSheet("background: transparent;")
        self.setCentralWidget(central)
        self._card = AdviceCard(central)
        # Initial position · overridden inside _align_to_mumu
        self._card.move(20, 20)

    def _setup_subscriber(self, ws_url: str):
        self._sub = AdviceSubscriber(ws_url)
        self._sub.advice_received.connect(self._on_advice)
        self._sub.connection_state.connect(self._on_state)
        self._sub.start()

    def _on_advice(self, payload: dict):
        log.info("advice · kind=%s · conf=%s", payload.get("kind"), payload.get("confidence"))
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
            # Degrade: MuMu not found / WSL · pin to the top-right of the screen
            screen_geo = QApplication.primaryScreen().geometry()
            w = AdviceCard.CARD_W + 40
            h = AdviceCard.CARD_H + 40
            self.setGeometry(screen_geo.width() - w - 20, 40, w, h)
            self._card.move(20, 20)
            return
        # Pin inside MuMu's top-right · overlay geometry = MuMu geometry · card biased to the top-right
        self.setGeometry(rect.left, rect.top, rect.width, rect.height)
        card_x = rect.width - AdviceCard.CARD_W - 24
        card_y = 24
        self._card.move(card_x, card_y)


# ==================== CLI ====================

def main():
    ap = argparse.ArgumentParser(
        description="jcc-coach overlay · a translucent advice card floating over the MuMu emulator",
    )
    ap.add_argument(
        "--ws-url",
        default="ws://localhost:8765/ws/advice",
        help="WebSocket address of the advice_server",
    )
    ap.add_argument(
        "--no-click-through",
        action="store_true",
        help="let the overlay receive mouse events (for debugging · click-through by default)",
    )
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
