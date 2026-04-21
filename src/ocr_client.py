"""PaddleOCR 封装 · 读屏幕中文文字（HP / 金币 / 等级数字 · 英雄名 · UI 气泡）。

为什么不让 VLM 直接读：
  实测 Qwen2.5-VL-7B 读 HP 经常错 1-2 · 对分析精度致命。
  专精 OCR 在中文数字/短文本上稳定 99%+ · 让 VLM 只做语义识别。
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

# ⚠ 关闭 onednn · paddle 3.x PIR + mkldnn 有 ConvertPirAttribute 未实现 bug
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_OCR = None  # 单例 · PaddleOCR 初始化耗时


def _get_ocr():
    global _OCR
    if _OCR is not None:
        return _OCR
    from paddleocr import PaddleOCR  # lazy import

    # PaddleOCR 3.x / 2.x 多版本兼容兜底
    kwargs_tries = [
        dict(use_textline_orientation=False, lang="ch", device="cpu",
             enable_mkldnn=False, cpu_threads=4),
        dict(use_textline_orientation=False, lang="ch", device="cpu"),
        dict(use_angle_cls=False, lang="ch", show_log=False, use_gpu=False,
             enable_mkldnn=False, cpu_threads=4),
        dict(use_angle_cls=False, lang="ch", show_log=False, use_gpu=False),
        dict(lang="ch"),
    ]
    last_err = None
    for kw in kwargs_tries:
        try:
            _OCR = PaddleOCR(**kw)
            log.info("PaddleOCR init OK with %s", list(kw.keys()))
            return _OCR
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"PaddleOCR init failed with all kwargs: {last_err}")


def recognize(image_bytes: bytes) -> list[dict]:
    """识别图中所有中文文字。返回 [{text, bbox: [x1,y1,x2,y2], conf}, ...]"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
    except Exception as e:
        log.warning("OCR: read image failed: %s", e)
        return []

    try:
        ocr = _get_ocr()
    except Exception as e:
        log.warning("OCR: init failed: %s", e)
        return []

    try:
        res = ocr.predict(arr) if hasattr(ocr, "predict") else ocr.ocr(arr, cls=False)
    except Exception as e:
        log.warning("OCR: predict failed: %s", e)
        return []

    out: list[dict] = []
    if not res:
        return out

    try:
        item = res[0] if isinstance(res, list) else res
        # PaddleOCR 3.x: OCRResult 是 dict-like
        try:
            d = dict(item)
        except Exception:
            d = None

        if d and "rec_texts" in d:
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            boxes = d.get("rec_boxes")
            polys = d.get("rec_polys") or d.get("dt_polys") or []
            for i, (t, s) in enumerate(zip(texts, scores)):
                if boxes is not None and i < len(boxes):
                    bx = boxes[i]
                    bbox = [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])]
                elif i < len(polys):
                    p = polys[i]
                    xs = [pt[0] for pt in p]
                    ys = [pt[1] for pt in p]
                    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                else:
                    continue
                out.append({
                    "text": str(t),
                    "bbox": bbox,
                    "conf": float(s),
                })
        else:
            # PaddleOCR 2.x 兼容
            lines = item if isinstance(item, list) else res
            for line in lines:
                if not (isinstance(line, list) and len(line) >= 2):
                    continue
                poly = line[0]
                text_conf = line[1]
                if isinstance(text_conf, (tuple, list)) and len(text_conf) >= 2:
                    text, conf = text_conf[0], text_conf[1]
                else:
                    continue
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                out.append({
                    "text": str(text),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "conf": float(conf),
                })
    except Exception as e:
        log.warning("OCR: parse result failed: %s", e)

    return out


def find_number_near(image_bytes: bytes, anchor_keyword: str, max_dist: int = 200) -> Optional[int]:
    """在含 `anchor_keyword` 的文字附近找最近的数字。
    用于读 HP / 金币 / 等级 等 —— anchor 是标签文字 · 数字是其值。"""
    import re
    texts = recognize(image_bytes)
    if not texts:
        return None
    anchors = [t for t in texts if anchor_keyword in t["text"]]
    if not anchors:
        return None
    a = anchors[0]
    acx = (a["bbox"][0] + a["bbox"][2]) // 2
    acy = (a["bbox"][1] + a["bbox"][3]) // 2

    candidates = []
    for t in texts:
        m = re.search(r"\d+", t["text"])
        if not m:
            continue
        tcx = (t["bbox"][0] + t["bbox"][2]) // 2
        tcy = (t["bbox"][1] + t["bbox"][3]) // 2
        dist = ((tcx - acx) ** 2 + (tcy - acy) ** 2) ** 0.5
        if dist <= max_dist:
            candidates.append((dist, int(m.group())))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _is_mostly_chinese(s: str) -> bool:
    if not s:
        return False
    cn = sum(1 for c in s if "一" <= c <= "鿿")
    return cn >= len(s.strip()) * 0.5


def find_long_text_bubble(
    image_bytes: bytes, screen_h: int = 1456, min_chars: int = 8, min_width_px: int = 300,
) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """启发式找屏幕下半部的"长中文气泡"（解说/提示/对话）。

    返回 (合并后的气泡文字, 首块中心点 (x, y))。
    """
    texts = recognize(image_bytes)
    if not texts:
        return None, None

    texts = [t for t in texts if t["conf"] > 0.6]
    bottom_y = int(screen_h * 0.7)
    bubbles = [
        t for t in texts
        if t["bbox"][1] >= bottom_y
        and len(t["text"].strip()) >= min_chars
        and (t["bbox"][2] - t["bbox"][0]) >= min_width_px
        and _is_mostly_chinese(t["text"])
    ]
    if not bubbles:
        return None, None

    text = " / ".join(b["text"] for b in bubbles[:4])
    b0 = bubbles[0]["bbox"]
    cx = (b0[0] + b0[2]) // 2
    cy = (b0[1] + b0[3]) // 2
    return text, (cx, cy)
