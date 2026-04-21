"""OpenCV 找教程亮色箭头中心。

金铲铲之战新手教程引导箭头特点：
- 颜色：亮黄 / 亮绿 / 亮白 大面积色块（箭头 + 高亮圈）
- 大小：一般 > 500 像素 · 不超过屏幕 1/4
- 位置：随机但通常在 UI 元素附近
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
    """找教程引导亮色箭头中心 (x, y) · 失败返回 None。"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
    except Exception as e:
        log.warning("arrow_finder: read image failed: %s", e)
        return None

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # 亮黄 · 教程箭头最常见色
    yellow_mask = cv2.inRange(hsv, np.array([20, 140, 180]), np.array([35, 255, 255]))
    # 亮绿
    green_mask = cv2.inRange(hsv, np.array([40, 140, 180]), np.array([80, 255, 255]))
    # 亮白（高亮圈）· 低饱和度 + 高亮度
    white_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 40, 255]))

    mask = cv2.bitwise_or(cv2.bitwise_or(yellow_mask, green_mask), white_mask)

    # 形态学开运算去细小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 连通域 · 过滤大小
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = 500
    max_area = h * w // 6
    valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    if not valid:
        return None

    # 取最大的连通块（箭头通常最大）
    c = max(valid, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_all_highlights(image_bytes: bytes, min_area: int = 500) -> list[dict]:
    """找所有亮色候选 · 调试用。返回 [{center, area, color}, ...]"""
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
