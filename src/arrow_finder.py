"""OpenCV: locate the center of the bright tutorial guide arrow.

Characteristics of the TFT (jkchess) new-player tutorial guide arrow:
- Color: bright yellow / bright green / bright white, large solid blocks (arrow + highlight ring)
- Size: usually > 500 pixels · no more than 1/4 of the screen
- Position: random, but usually near a UI element
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def find_arrow(image_bytes: bytes) -> Optional[tuple[int, int]]:
    """Find the center (x, y) of the bright tutorial guide arrow · returns None on failure."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
    except Exception as e:
        log.warning("arrow_finder: read image failed: %s", e)
        return None

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # Bright yellow · the most common tutorial-arrow color
    yellow_mask = cv2.inRange(hsv, np.array([20, 140, 180]), np.array([35, 255, 255]))
    # Bright green
    green_mask = cv2.inRange(hsv, np.array([40, 140, 180]), np.array([80, 255, 255]))
    # Bright white (highlight ring) · low saturation + high brightness
    white_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 40, 255]))

    mask = cv2.bitwise_or(cv2.bitwise_or(yellow_mask, green_mask), white_mask)

    # Morphological opening to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Connected components · filter by size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = 500
    max_area = h * w // 6
    valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    if not valid:
        return None

    # Take the largest connected block (the arrow is usually the biggest)
    c = max(valid, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_all_highlights(image_bytes: bytes, min_area: int = 500) -> list[dict]:
    """Find all bright-color candidates · for debugging. Returns [{center, area, color}, ...]"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
    except Exception:
        return []

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    max_area = h * w // 6

    results = []
    color_specs = {
        "yellow": ([20, 140, 180], [35, 255, 255]),
        "green":  ([40, 140, 180], [80, 255, 255]),
        "white":  ([0, 0, 220],   [180, 40, 255]),
    }
    for color, (lo, hi) in color_specs.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            results.append({"center": (cx, cy), "area": int(area), "color": color})
    results.sort(key=lambda r: -r["area"])
    return results
