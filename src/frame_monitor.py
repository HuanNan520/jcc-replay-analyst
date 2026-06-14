"""The "screen-change sentinel" of the event-driven main loop.

Principle: compute a dHash per ROI (region of interest), compare each frame to the previous
      one by Hamming distance, and fire an event when the distance exceeds the threshold.
      This is 10-20x cheaper on VRAM than hammering the VLM every second.

Design principles:
- Zero external dependencies (Pillow only, already in the venv)
- ROIs use normalized coordinates (0-1), locked to actual screen pixels at construction time
- Each ROI has its own threshold: the combat board jitters a lot (high threshold), HUD numbers change little (low threshold)
- Event classification is left to the upper layer; this module only produces the raw event stream
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

from PIL import Image


# =============================================================================
# Perceptual hash (dHash) — zero-dependency implementation
# =============================================================================

HASH_SIZE = 8  # produces a 64-bit fingerprint


def dhash(img: Image.Image, hash_size: int = HASH_SIZE) -> int:
    """Difference hash.
    image -> grayscale -> resize to (hash_size+1, hash_size) -> adjacent-pixel diff -> 64-bit integer.
    Robust to lighting/compression distortion; sensitive to real content changes.
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
    """Hamming distance: how many bits differ."""
    return bin(a ^ b).count("1")


# =============================================================================
# Default ROI config — TFT portrait layout
# =============================================================================

# (x, y, w, h) normalized coordinates. height > width = portrait. Use the other set
# for landscape, or rotate the frame before passing it in.
DEFAULT_REGIONS_PORTRAIT: dict[str, tuple[float, float, float, float]] = {
    "hud_top":      (0.00, 0.00, 1.00, 0.10),  # top: HP / gold / round / countdown
    "carry_zone":   (0.00, 0.10, 1.00, 0.55),  # main board area (combat plays out here)
    "bench_row":    (0.00, 0.65, 1.00, 0.10),  # bench row + item slots
    "shop_bottom":  (0.00, 0.75, 1.00, 0.20),  # bottom 5 shop cards
    "center_popup": (0.10, 0.20, 0.80, 0.55),  # centered popup (augment / draft / settlement / win-loss)
}

# Per-ROI Hamming-distance threshold (>= counts as a "change")
DEFAULT_THRESHOLDS: dict[str, int] = {
    "hud_top":      6,   # numbers changing must trigger, so a low threshold
    "shop_bottom":  6,
    "carry_zone":   20,  # combat animation jitters heavily, high threshold to avoid false positives
    "bench_row":    6,
    "center_popup": 8,   # catch both popup appearance and dismissal
}

# Landscape-layout normalized ROIs (calibrated from measured 2560x1456 screenshots)
# Data source: data/screens/session1/ full S16 match, 2026-04-21
DEFAULT_REGIONS_LANDSCAPE: dict[str, tuple[float, float, float, float]] = {
    "hud_top":      (0.00, 0.00, 1.00, 0.07),  # top: round / gold numbers / opponent-avatar strip
    "trait_left":   (0.00, 0.05, 0.08, 0.75),  # left-side trait bar (Piltover / Guardian etc.)
    "carry_zone":   (0.08, 0.10, 0.65, 0.65),  # board combat area
    "bench_row":    (0.18, 0.75, 0.60, 0.08),  # 9-slot bench row
    "shop_bottom":  (0.25, 0.88, 0.55, 0.10),  # bottom shop, 5 cards
    "right_panel":  (0.90, 0.05, 0.10, 0.80),  # right-side opponent preview
    "center_popup": (0.22, 0.22, 0.56, 0.50),  # center popup (augment / settlement / draft)
}

DEFAULT_THRESHOLDS_LANDSCAPE: dict[str, int] = {
    "hud_top":      5,    # sensitive to round / gold number changes
    "trait_left":   5,    # sensitive to trait-activation changes
    "carry_zone":   22,   # large combat animation, high threshold to avoid false positives
    "bench_row":    6,
    "shop_bottom":  6,    # must catch shop refreshes
    "right_panel":  6,    # opponent HP changes
    "center_popup": 10,   # only catch popup appearance
}


# =============================================================================
# Data structures
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
    """Manages multiple ROIs, emitting an event list per frame."""

    def __init__(
        self,
        screen_size: tuple[int, int],
        regions: Mapping[str, tuple[float, float, float, float]] | None = None,
        thresholds: Mapping[str, int] | None = None,
        orientation: str = "auto",
    ):
        """
        Args:
            screen_size: (width, height) in screen pixels.
            regions: ROI normalized coordinates. When None, picks a default by orientation.
            thresholds: per-region Hamming-distance thresholds. None as above.
            orientation: "portrait" / "landscape" / "auto" (default: decided by w vs h)
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
        """Normalized coordinates -> the (left, top, right, bottom) pixels PIL crop needs."""
        x, y, w, h = self.regions[name]
        return (
            int(x * self.screen_w),
            int(y * self.screen_h),
            int((x + w) * self.screen_w),
            int((y + h) * self.screen_h),
        )

    def observe(self, img_source: ImageSource) -> list[FrameEvent]:
        """Take one screenshot, return an event per ROI.
        On the first frame (no baseline), everything returns triggered=False and the baseline is established.
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
                # First sighting, establish the baseline
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
        """Clear the baseline so the next frame starts fresh (use on scene transitions)."""
        self._last_hashes.clear()
        self.frame_count = 0


# =============================================================================
# Coarse-grained event classification (optional, for the upper layer)
# =============================================================================

def classify(changed_regions: Sequence[str]) -> str:
    """Infer a coarse event type from which ROIs changed.
    Fine-grained judgment (e.g. "shop refresh" vs "shop changed after buying a card") is left to the VLM.

    Heuristics (strict):
      - A real popup requires center_popup + shop_bottom + hud_top to change together (a popup covers the full screen)
      - Only center_popup changing = combat / unit animation, not a popup
      - carry_zone changing alone = mid-combat, no decision needed
    """
    regs = set(changed_regions)
    if not regs:
        return "idle"

    # Real popup: a popup half-covers the screen -> center_popup changes a lot + at least 2 major ROIs also change
    # (the augment screen covers both shop_bottom and part of hud_top)
    major_regs = regs & {"hud_top", "shop_bottom", "right_panel", "trait_left"}
    if "center_popup" in regs and len(major_regs) >= 2:
        return "popup"

    # Shop + HUD changing together = a transaction (buy/sell)
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
        return "board_motion"       # combat animation; the main loop should ignore it and not trigger a decision
    return "unknown"


# =============================================================================
# Convenience entry point
# =============================================================================

def monitor_from_first_screenshot(
    img_source: ImageSource,
    **kwargs,
) -> FrameMonitor:
    """Build a monitor directly from the first screenshot, inferring resolution. Saves passing screen_size manually."""
    img = FrameMonitor._load_image(img_source)
    mon = FrameMonitor(screen_size=img.size, **kwargs)
    mon.observe(img)  # use this frame as the baseline
    return mon
