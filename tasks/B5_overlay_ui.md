# B5 · PyQt 桌面 overlay

**分配给**：Claude Opus 4.7（`claude-opus-4-7`）· 三合一工程（PyQt 框架 + Win32 窗口对齐 + WebSocket 订阅）· 且有 UI 设计感要求。
**依赖**：B4（WebSocket 服务先跑起来）。
**预期工时**：1 天（含视觉打磨）。
**运行时**：**Windows 原生 Python**（WSL 跑 PyQt GUI 能跑但窗口对齐 MuMu 需要 Win32 API）。
**新产品定位中的角色**：**玩家看得到的那一层** —— 产品门面 · 作品集 demo 视频的主角。

---

## 你是谁

你是被派到 `HuanNan520/jcc-replay-analyst` 执行 B5 的 Claude Opus 4.7。
B4 已经 merge · 你订阅它的 WebSocket 拿 advice 流。你的任务：**写一个半透明悬浮窗 · 覆盖在 MuMu 模拟器窗口上方 · 把 advice 展示给玩家**。

这是产品的视觉门面 —— 演示视频第一眼看到的就是你写的 overlay · 视觉质量直接影响作品集效果。

## 目标视觉

```
┌─────────────────────── MuMu 游戏窗口 ──────────────────────────────┐
│                                                                     │
│   [游戏画面]                                     ┌────────────┐     │
│                                                  │ ★ 选增强    │     │
│                                                  │ 选第 1 个   │     │
│                                                  │ 法师之力     │     │
│                                                  │ 契合度 88%  │     │
│                                                  │ 推荐理由... │     │
│                                                  └────────────┘     │
│                                                  （300×180 半透明  │
│                                                   卡片 · 金色边框 · │
│                                                   淡入淡出）         │
└─────────────────────────────────────────────────────────────────────┘
```

关键特性：
- **Frameless + 置顶 + 半透明背景**
- **点击穿透**（鼠标事件不被 overlay 拦截 · 玩家照常操作游戏）
- **跟随 MuMu 窗口移动**（Win32 FindWindow + GetWindowRect · poll 每 500ms）
- **金色边框 + 宋体标题 + 无衬线正文**（延续项目视觉语言）
- **淡入淡出**（advice 来临 fade-in · 8 秒后 fade-out · 或被新 advice 替换）

---

## 具体要做

### 1. 新增 `src/overlay_ui.py`

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
    """Windows 原生 · 找 MuMu 窗口坐标。WSL / Linux 返回 None。"""
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
    # 取面积最大的（主窗口）
    return max(found, key=lambda r: r.width * r.height)


# ==================== WebSocket client (qasync-free 版本) ====================

class AdviceSubscriber(QObject):
    """后台线程跑 websockets · emit Qt signal 到主线程。"""

    advice_received = pyqtSignal(dict)   # 带 payload dict
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
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("type") in ("advice", "history"):
                            self.advice_received.emit(msg["payload"])
            except Exception as e:
                log.warning("WS 断线 · 5s 后重连 · %s", e)
                self.connection_state.emit("disconnected")
                await asyncio.sleep(5)


# ==================== Advice Card Widget ====================

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
        # 半透明深色底
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor(19, 17, 28, 235))
        bg.setColorAt(1, QColor(11, 9, 18, 235))
        p.fillRect(self.rect(), QBrush(bg))
        # 金色边框
        pen = QPen(self._accent)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        # 顶部 accent 短线
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
        # 默认放右上角
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
            # MuMu 没找到 · 贴右上角屏幕
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen.width() - 360, 40, 360, 220)
            self._card.move(20, 20)
            return
        # 贴到 MuMu 右上内部
        self.setGeometry(rect.left, rect.top, rect.width, rect.height)
        card_x = rect.width - self._card.width() - 24
        card_y = 20
        self._card.move(card_x, card_y)


# ==================== CLI ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default="ws://localhost:8765/ws/advice")
    ap.add_argument("--no-click-through", action="store_true", help="overlay 可接收鼠标（便于调试）")
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

### 2. 更新 `requirements-windows.txt`

```
PyQt6>=6.7
websockets>=12.0
```

（不加到主 requirements.txt · 因为 PyQt6 在 Linux headless 下装得慢没必要）

### 3. 单元测试 `tests/test_overlay_headless.py`

PyQt 不好在 headless CI 测 · 只测纯函数：

```python
import pytest


def test_find_mumu_rect_on_linux_returns_none():
    """WSL / Linux 下 user32 不可用 · 应优雅返回 None。"""
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

CI 里这个文件在 Linux 可以跑（user32 is None · test_find_mumu_rect_on_linux 只断言 None），但 PyQt6 import 要不装就 import 时跳过。用 `pytest.importorskip("PyQt6", reason="Windows UI")` 在文件顶部。

**更好**：在 `tests/test_overlay_headless.py` 顶部：
```python
import pytest
pytest.importorskip("PyQt6", reason="Windows UI only")
```

### 4. 更新 `.github/workflows/ci.yml`

在 pip install 那步里显式跳过 PyQt6（别被 requirements 间接拖进来）：

```yaml
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pydantic httpx pillow numpy pytest fastapi uvicorn websockets
          # PyQt6 Windows-only · CI 不装
      - name: Run tests
        run: |
          pytest tests/ -v --ignore=tests/test_overlay_headless.py
```

或者保留 overlay 测试但依赖 pytest.importorskip 自动跳过 · 都行。选一个。

### 5. README 最后一段实时 coach 启动完整流程

```markdown
### 实时 coach 模式 · 完整启动（4 个终端）

```powershell
# 终端 1 · WSL · 启 vLLM
source ~/jcc-replay-analyst/.venv/bin/activate
python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-VL-4B-FP8 --port 8000

# 终端 2 · Windows 或 WSL · 启 advice server
python -m src.advice_server --port 8765

# 终端 3 · Windows 原生 Python · 起 OBS virtual cam + live tick（需前置 OBS Start Virtual Camera）
python -m src.live_tick --fps 2 --advice-server http://localhost:8765

# 终端 4 · Windows 原生 Python · 起 overlay
python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice
```

玩金铲铲 · overlay 会自动浮在 MuMu 窗口上方 · 决策点触发就弹建议。
```

---

## 禁止做的事

- 不要引入 Electron / Tauri / web UI 框架 · 就 PyQt6
- 不要自己写 WebSocket reconnect 指数退避 —— 5s 固定间隔够
- 不要加配置文件 / YAML · CLI 参数够
- 不要做"overlay 内的配置面板" · v1 只管展示
- 不要处理多 MuMu 实例 · 取面积最大的那个
- 不要做截屏 / 录制功能 · 跟实时 coach 无关
- 不要改 B4 `src/advice_server.py` —— 有新需求告诉用户别改 server
- 不要改 `src/schema.py` · 其它 src/ 文件

---

## 自验收清单

Linux/WSL 侧（CI 能跑的）：
- [ ] `python -c "from src.overlay_ui import find_mumu_rect, WindowRect, KIND_DISPLAY"` 无错
- [ ] `pytest tests/test_overlay_headless.py -v` 全绿（Linux 下 find_mumu_rect 测试应得 None）
- [ ] `pytest tests/ -v --ignore=tests/test_overlay_headless.py` 原有 40+ 测试零回归

Windows 侧（用户机器实测 · 你至少要给用户贴命令让他验）：
- [ ] `pip install -r requirements-windows.txt` 无错
- [ ] advice_server 跑着时 · `python -m src.overlay_ui` 能启动 · 窗口透明看到桌面
- [ ] 开着 MuMu · overlay 窗口**自动**贴到 MuMu 上（坐标对齐）
- [ ] 用 `curl -X POST http://localhost:8765/advice -d '{...一个合法 advice...}'` 手动推 · overlay 右上角**淡入**显示卡片 · 8 秒淡出
- [ ] overlay 不拦截鼠标（你在 overlay 上点击 · 鼠标事件到 MuMu 窗口）—— 如果用 `--no-click-through` 调试模式可以点到 overlay 本身
- [ ] 关掉 advice_server · overlay 不崩 · log 出 "WS 断线 · 5s 后重连"

`git diff --stat` 只含：
- `src/overlay_ui.py` (新)
- `tests/test_overlay_headless.py` (新)
- `requirements-windows.txt` (追加)
- `.github/workflows/ci.yml` (可选改 · 显式跳过 PyQt6)
- `README.md` (加一段)

## 完成后

给用户 ≤ 200 字报告：
- overlay 截图（存 `/tmp/overlay_demo.png` 或让他手动截）· 至少描述视觉效果
- MuMu 窗口坐标对齐表现（跟随移动流畅否）
- 每个 kind 的卡片长什么样（是否 6 类都 UI 兼容）
- 用户桌面环境适配情况（分辨率 · 缩放 · 多显示器有无问题）
- 后续调优 TODO（比如动画 · 多卡片堆叠 · 自适应字号）

不 git commit。

---

## 参考

- PyQt6 Window Flags: https://doc.qt.io/qt-6/qt.html#WindowType-enum
- WA_TransparentForMouseEvents 点击穿透: https://doc.qt.io/qt-6/qt.html#WidgetAttribute-enum
- Win32 EnumWindows + GetWindowRect: ctypes 直打 user32.dll · 不装 pywin32 · 减少依赖
- 字体栈：延续 pitch/index.html 的 Songti SC / Baskerville / PingFang SC 组合
