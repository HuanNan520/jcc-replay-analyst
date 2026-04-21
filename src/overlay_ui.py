"""PyQt 桌面 overlay · 半透明卡片浮在 MuMu 模拟器窗口上方。

B5 · 产品门面 —— 玩家看得到的那一层 · 订阅 B4 的 WebSocket 建议流。

关键特性：
- Frameless + 置顶 + 半透明背景 · WA_TranslucentBackground
- 点击穿透（WA_TransparentForMouseEvents）· 玩家照常操作游戏
- 跟随 MuMu 窗口移动（Win32 FindWindow + GetWindowRect · poll 500ms）
- 金色边框 + 宋体标题 + 无衬线正文 · 延续 pitch/index.html 视觉语言
- 淡入淡出（400ms OutCubic / 600ms InCubic）· 新 advice 替换旧 · 8s 后自动淡出
- WebSocket 重连 5s 固定间隔 · server 断开不崩

运行时（Windows 原生 Python）：
    pip install -r requirements-windows.txt
    python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice

WSL / Linux 下 PyQt6 能装但 MuMu 窗口对齐走 Win32 · find_mumu_rect 返回 None
→ 降级为 overlay 贴右上角屏幕（方便无 MuMu 开发调试）。
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
    # WSL / Linux · 没 user32.dll · 降级
    user32 = None


@dataclass
class WindowRect:
    """Win32 窗口矩形 · 用 dataclass 方便测试。"""
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
    title_contains: tuple[str, ...] = ("MuMu", "模拟器"),
) -> Optional[WindowRect]:
    """Windows 原生 · 找 MuMu 窗口坐标。

    WSL / Linux 下 user32 is None · 返回 None → 调用方降级为屏幕右上角。
    多实例取面积最大的那个（主窗口）· 不处理多开场景（v1 约定）。
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
    """后台线程跑 websockets · emit Qt signal 到主线程。

    不用 qasync（多一个依赖）· 开一个 daemon thread 自跑 asyncio。
    断线 5s 固定间隔重连 · 不上指数退避（server 同机 · 简单够用）。
    """

    advice_received = pyqtSignal(dict)   # payload dict · 来自 history 或 advice 消息
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
            log.error("websockets 未安装 · pip install websockets")
            self.connection_state.emit("error")
            return

        while not self._stop:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.connection_state.emit("connected")
                    log.info("WS 已连接 · %s", self.ws_url)
                    async for raw in ws:
                        if raw == "pong":
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            log.debug("WS 收到非 JSON · 忽略 · %r", raw[:60])
                            continue
                        if msg.get("type") in ("advice", "history"):
                            self.advice_received.emit(msg["payload"])
            except Exception as e:
                log.warning("WS 断线 · 5s 后重连 · %s", e)
                self.connection_state.emit("disconnected")
                await asyncio.sleep(5)


# ==================== Advice Card Widget ====================

# 六类决策 · 每类独立颜色 + 符号 · 延续 pitch/index.html 视觉
KIND_DISPLAY = {
    "augment":     ("★ 选增强",    "#e6c17a"),  # 金
    "carousel":    ("⚫ 轮抱",      "#c9a45d"),  # 暖金
    "shop":        ("◆ 商店",      "#5a8b7a"),  # 青绿
    "level":       ("▲ 升级决策",  "#b3432e"),  # 赤朱
    "positioning": ("◈ 摆位",      "#9c7a3c"),  # 深金
    "item":        ("✦ 装备",      "#c9a45d"),  # 暖金
}


class AdviceCard(QWidget):
    """单张半透明卡片 · 300x180 · 金色边框 · 淡入淡出动画。"""

    CARD_W = 320
    CARD_H = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.CARD_W, self.CARD_H)

        # 透明度效果 · 用于淡入淡出
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        # kind 标签 · 宋体中号
        self._kind_label = QLabel("")
        f1 = QFont("Songti SC", 12)
        f1.setWeight(QFont.Weight.Medium)
        self._kind_label.setFont(f1)

        # 推荐主文案 · 宋体大号
        self._rec_label = QLabel("")
        f2 = QFont("Songti SC", 18)
        f2.setWeight(QFont.Weight.Normal)
        self._rec_label.setFont(f2)
        self._rec_label.setWordWrap(True)
        self._rec_label.setStyleSheet("color: #f0e4c8;")

        # 推理理由 · 无衬线小号
        self._reason_label = QLabel("")
        f3 = QFont("PingFang SC", 10)
        self._reason_label.setFont(f3)
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #a39d8e;")

        # 置信度 · Baskerville 斜体
        self._conf_label = QLabel("")
        f4 = QFont("Baskerville", 9)
        f4.setItalic(True)
        self._conf_label.setFont(f4)
        self._conf_label.setStyleSheet("color: #6b6458;")

        layout.addWidget(self._kind_label)
        layout.addWidget(self._rec_label)
        layout.addWidget(self._reason_label, 1)
        layout.addWidget(self._conf_label)

        # 动画
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

        # 8 秒自动淡出定时器
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out.start)

        self._accent = QColor("#e6c17a")

    def show_advice(self, payload: dict, display_ms: int = 8000) -> None:
        """收到新 advice · 填充卡片 · 启动淡入 · 设置 8s 淡出。

        字段兼容：
          - kind 必填 · 不在 KIND_DISPLAY 时降级为 "◇ {kind}"
          - recommendation / action 二选一 · 都没有显示 "—"
          - reasoning 可空 · 截断到 160 字（卡片尺寸有限）
          - confidence 可空 · 默认 0
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

        # 重新触发动画 · 如果上一条还在显示会直接替换
        self._hide_timer.stop()
        self._fade_out.stop()
        self._fade_in.stop()
        self._fade_in.start()
        self._hide_timer.start(display_ms)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 半透明深色渐变背景
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor(19, 17, 28, 235))
        bg.setColorAt(1, QColor(11, 9, 18, 235))
        p.fillRect(self.rect(), QBrush(bg))

        # 金色边框（accent 色）
        pen = QPen(self._accent)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

        # 顶部 accent 短线 · 2px 高 60px 宽
        p.fillRect(QRect(0, 0, 60, 2), self._accent)


# ==================== Main Overlay Window ====================

class OverlayWindow(QMainWindow):
    """主 overlay 窗 · frameless + 置顶 + 透明 · 跟随 MuMu 坐标。"""

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
            | Qt.WindowType.Tool  # Tool 不占任务栏
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if self._click_through:
            # 点击穿透 · 鼠标事件到 overlay 下方的窗口（MuMu）
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _setup_ui(self):
        central = QWidget()
        central.setStyleSheet("background: transparent;")
        self.setCentralWidget(central)
        self._card = AdviceCard(central)
        # 初始位置 · _align_to_mumu 里会覆盖
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
            # 降级：MuMu 没找到 / WSL · 贴右上角屏幕
            screen_geo = QApplication.primaryScreen().geometry()
            w = AdviceCard.CARD_W + 40
            h = AdviceCard.CARD_H + 40
            self.setGeometry(screen_geo.width() - w - 20, 40, w, h)
            self._card.move(20, 20)
            return
        # 贴到 MuMu 右上内部 · overlay 几何 = MuMu 几何 · 卡片偏右上
        self.setGeometry(rect.left, rect.top, rect.width, rect.height)
        card_x = rect.width - AdviceCard.CARD_W - 24
        card_y = 24
        self._card.move(card_x, card_y)


# ==================== CLI ====================

def main():
    ap = argparse.ArgumentParser(
        description="jcc-coach overlay · 半透明 advice 卡片浮在 MuMu 模拟器上",
    )
    ap.add_argument(
        "--ws-url",
        default="ws://localhost:8765/ws/advice",
        help="advice_server 的 WebSocket 地址",
    )
    ap.add_argument(
        "--no-click-through",
        action="store_true",
        help="overlay 可接收鼠标（便于调试 · 默认点击穿透）",
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
