"""事件驱动主循环的"屏幕变化哨兵"。

原理：按 ROI（region of interest）计算 dHash，每帧和上帧比汉明距离，
      超阈值就触发事件。比每秒硬调 VLM 省 10-20 倍显存。

设计原则：
- 零外部依赖（只用 Pillow，venv 已装）
- ROI 用归一化坐标（0-1），实例化时锁定到实际屏幕像素
- 各 ROI 独立阈值：战斗棋盘抖动大（阈值高），HUD 数字变化小（阈值低）
- 事件分类交给上层，本模块只产原始事件流
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

from PIL import Image


# =============================================================================
# 感知哈希（dHash） — 零依赖实现
# =============================================================================

HASH_SIZE = 8  # 产出 64 bit 指纹


def dhash(img: Image.Image, hash_size: int = HASH_SIZE) -> int:
    """差分哈希。
    图像 → 灰度 → 缩至 (hash_size+1, hash_size) → 相邻像素差分 → 64bit 整数。
    对光照/压缩失真鲁棒；对内容真变化敏感。
    """
    g = img.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(g.getdata())
    value = 0
    for row in range(hash_size):
        base = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | (
                1 if pixels[base + col] > pixels[base + col + 1] else 0
            )
    return value


def hamming(a: int, b: int) -> int:
    """汉明距离：有多少 bit 不一样"""
    return bin(a ^ b).count("1")


# =============================================================================
# 默认 ROI 配置 —— 金铲铲竖屏画面
# =============================================================================

# (x, y, w, h) 归一化坐标。高 > 宽 = 竖屏。横屏用另一套或旋转后传入。
DEFAULT_REGIONS_PORTRAIT: dict[str, tuple[float, float, float, float]] = {
    "hud_top":      (0.00, 0.00, 1.00, 0.10),  # 顶部血量/金币/回合/倒计时
    "carry_zone":   (0.00, 0.10, 1.00, 0.55),  # 棋盘主区（战斗播放在这里）
    "bench_row":    (0.00, 0.65, 1.00, 0.10),  # 备战行 + 装备槽
    "shop_bottom":  (0.00, 0.75, 1.00, 0.20),  # 底部 5 张商店卡
    "center_popup": (0.10, 0.20, 0.80, 0.55),  # 居中弹窗（海克斯/选秀/结算/胜负）
}

# 各 ROI 汉明距离阈值（≥ 即视为"变化"）
DEFAULT_THRESHOLDS: dict[str, int] = {
    "hud_top":      6,   # 数字变了就要 trigger，阈值低
    "shop_bottom":  6,
    "carry_zone":   20,  # 战斗动画大幅抖动，阈值高防误报
    "bench_row":    6,
    "center_popup": 8,   # 弹窗出现/消失都要抓
}

# 横屏布局归一化 ROI（基于 2560×1456 实测截图标定）
# 数据源：data/screens/session1/ 完整 S16 对局，2026-04-21
DEFAULT_REGIONS_LANDSCAPE: dict[str, tuple[float, float, float, float]] = {
    "hud_top":      (0.00, 0.00, 1.00, 0.07),  # 顶部：回合/金币数字/对手头像一排
    "trait_left":   (0.00, 0.05, 0.08, 0.75),  # 左侧羁绊栏（皮尔特沃夫/护卫等）
    "carry_zone":   (0.08, 0.10, 0.65, 0.65),  # 棋盘战斗区
    "bench_row":    (0.18, 0.75, 0.60, 0.08),  # 备战行 9 格
    "shop_bottom":  (0.25, 0.88, 0.55, 0.10),  # 底部商店 5 卡
    "right_panel":  (0.90, 0.05, 0.10, 0.80),  # 右侧对手预览
    "center_popup": (0.22, 0.22, 0.56, 0.50),  # 中央弹窗（augment/结算/选秀）
}

DEFAULT_THRESHOLDS_LANDSCAPE: dict[str, int] = {
    "hud_top":      5,    # 回合/金币数字变化敏感
    "trait_left":   5,    # 羁绊激活变化敏感
    "carry_zone":   22,   # 战斗动画大，阈值高防误报
    "bench_row":    6,
    "shop_bottom":  6,    # 商店刷新要抓
    "right_panel":  6,    # 对手血量变
    "center_popup": 10,   # 弹窗出现才抓
}


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class FrameEvent:
    region: str
    distance: int
    triggered: bool
    timestamp: float


ImageSource = Union[bytes, bytearray, Image.Image]


# =============================================================================
# FrameMonitor
# =============================================================================

class FrameMonitor:
    """管理多个 ROI，每帧输出事件列表。"""

    def __init__(
        self,
        screen_size: tuple[int, int],
        regions: Mapping[str, tuple[float, float, float, float]] | None = None,
        thresholds: Mapping[str, int] | None = None,
        orientation: str = "auto",
    ):
        """
        Args:
            screen_size: (width, height) 屏幕像素。
            regions: ROI 归一化坐标。None 时按 orientation 选默认。
            thresholds: 各 region 汉明距离阈值。None 同上。
            orientation: "portrait" / "landscape" / "auto"（默认按 w vs h 判断）
        """
        self.screen_w, self.screen_h = screen_size
        if orientation == "auto":
            orientation = "landscape" if self.screen_w > self.screen_h else "portrait"
        self.orientation = orientation

        if regions is None:
            regions = (DEFAULT_REGIONS_LANDSCAPE if orientation == "landscape"
                       else DEFAULT_REGIONS_PORTRAIT)
        if thresholds is None:
            thresholds = (DEFAULT_THRESHOLDS_LANDSCAPE if orientation == "landscape"
                          else DEFAULT_THRESHOLDS)

        self.regions = dict(regions)
        self.thresholds = dict(thresholds)
        self._last_hashes: dict[str, int] = {}
        self.frame_count = 0

    def _crop_box_px(self, name: str) -> tuple[int, int, int, int]:
        """归一化坐标 → PIL crop 所需的 (left, top, right, bottom) 像素。"""
        x, y, w, h = self.regions[name]
        return (
            int(x * self.screen_w),
            int(y * self.screen_h),
            int((x + w) * self.screen_w),
            int((y + h) * self.screen_h),
        )

    def observe(self, img_source: ImageSource) -> list[FrameEvent]:
        """吃一张截图，返回每个 ROI 的事件。
        首帧（没有 baseline）全部返回 triggered=False 并建立 baseline。
        """
        img = self._load_image(img_source)
        self.frame_count += 1
        now = time.time()
        events: list[FrameEvent] = []

        for name in self.regions:
            box = self._crop_box_px(name)
            crop = img.crop(box)
            cur = dhash(crop, HASH_SIZE)
            last = self._last_hashes.get(name)

            if last is None:
                # 首次见，建立 baseline
                self._last_hashes[name] = cur
                events.append(FrameEvent(name, 0, False, now))
                continue

            dist = hamming(cur, last)
            threshold = self.thresholds.get(name, 5)
            triggered = dist >= threshold
            self._last_hashes[name] = cur
            events.append(FrameEvent(name, dist, triggered, now))

        return events

    @staticmethod
    def _load_image(src: ImageSource) -> Image.Image:
        if isinstance(src, Image.Image):
            return src
        if isinstance(src, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(src)))
        raise TypeError(
            f"img_source must be bytes or PIL.Image, got {type(src).__name__}"
        )

    def any_triggered(self, events: Sequence[FrameEvent]) -> bool:
        return any(e.triggered for e in events)

    def changed_regions(self, events: Sequence[FrameEvent]) -> list[str]:
        return [e.region for e in events if e.triggered]

    def reset(self) -> None:
        """清空 baseline，下一帧重新起点（场景切换时用）"""
        self._last_hashes.clear()
        self.frame_count = 0


# =============================================================================
# 粗粒度事件分类（上层可选用）
# =============================================================================

def classify(changed_regions: Sequence[str]) -> str:
    """根据哪些 ROI 变了，推一个粗事件类型。
    精细判断（比如"商店刷新" vs "买卡后商店变"）交给 VLM 解析。

    启发式（严格）：
      - 真 popup 要求 center_popup + shop_bottom + hud_top 同时变（弹窗全屏遮盖）
      - 只 center_popup 变 = 战斗/棋子动画，不是弹窗
      - carry_zone 单独变 = 战斗中，不用决策
    """
    regs = set(changed_regions)
    if not regs:
        return "idle"

    # 真正 popup：弹窗会半覆盖屏幕 → center_popup 大变 + 至少 2 个主 ROI 也变
    # （augment 界面会同时遮掉 shop_bottom 和部分 hud_top）
    major_regs = regs & {"hud_top", "shop_bottom", "right_panel", "trait_left"}
    if "center_popup" in regs and len(major_regs) >= 2:
        return "popup"

    # 商店 + HUD 同时变 = 交易（买/卖）
    if "shop_bottom" in regs and "hud_top" in regs:
        return "trade"
    if "shop_bottom" in regs:
        return "shop_refresh"
    if "hud_top" in regs:
        return "hud_change"
    if "right_panel" in regs:
        return "right_panel_change"
    if "bench_row" in regs:
        return "bench_change"
    if "carry_zone" in regs or "center_popup" in regs:
        return "board_motion"       # 战斗动画；主循环应当忽略，不触发决策
    return "unknown"


# =============================================================================
# 便捷入口
# =============================================================================

def monitor_from_first_screenshot(
    img_source: ImageSource,
    **kwargs,
) -> FrameMonitor:
    """从第一张截图直接推出分辨率建 monitor。省得手动传 screen_size。"""
    img = FrameMonitor._load_image(img_source)
    mon = FrameMonitor(screen_size=img.size, **kwargs)
    mon.observe(img)  # 把这张作为 baseline
    return mon
