"""Qwen VLM 客户端 —— 语义识别层。

职责：读一帧画面的语义字段 —— 羁绊名 · 阵容 · 棋子分布 · 增强符文 —— 返回 WorldState。
数字类（HP / 金币 / 等级）交给 OCR · 小图标交给 CV · VLM 只做它擅长的。

接入：
  vLLM 的 OpenAI 兼容接口（Qwen2.5-VL / Qwen3-VL 都可）· 或任何 OpenAI-compat endpoint。
  mode="mock" 不依赖服务 · 返回一个合法的 WorldState（便于调试 pipeline）。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional

import httpx

from .schema import (
    ActiveTrait, BagItem, OpponentPreview, Unit, WorldState,
)

log = logging.getLogger(__name__)


VLM_PARSE_PROMPT = """你是《金铲铲之战》画面识别器。分析这张横屏截图 · 输出严格的 JSON。

必填字段：
  stage           "pick"/"pve"/"pvp"/"augment"/"carousel"/"item"/"positioning"/"end"/"unknown"
  round           "3-2" 这种格式 · 如不可见填 "?-?"
  hp              0-100 · 不可见填 0
  gold            >=0 · 不可见填 0
  level           1-10 · 不可见填 1
  exp             "x/y" 或 "0/0"
  board           场上棋子列表 [{name, star, items, position:[row,col]}]
  bench           备战席棋子 [{name, star, items, position:null}]
  bag             散装备组件 [{slot, name}] · 组件只能是：暴风大剑/反曲之弓/无用大棒/锁子甲/负极斗篷/巨人腰带/女神之泪/暴风之刃/训练假人
  shop            商店 5 个可见棋子名 []
  active_traits   激活羁绊 [{name, count, tier: bronze/silver/gold/prismatic}]
  augments        已选增强名
  opponents_preview 其他玩家血量预览 [{hp, top_carry, comp_summary}]

严格只输出 JSON · 不要 markdown · 不要解释。
"""


def _coerce_world_state(payload: dict) -> WorldState:
    """把 VLM 松散输出强行规范成合法 WorldState · 失败字段给默认值。"""
    def _s(v, default=""): return str(v) if v is not None else default
    def _i(v, default=0, lo=None, hi=None):
        try:
            r = int(v)
        except Exception:
            return default
        if lo is not None and r < lo: r = lo
        if hi is not None and r > hi: r = hi
        return r

    stage = payload.get("stage", "unknown")
    if stage not in {"pick", "pve", "pvp", "augment", "carousel", "item", "positioning", "end", "unknown"}:
        stage = "unknown"

    def _coerce_unit(u: dict) -> Optional[Unit]:
        try:
            name = _s(u.get("name"))
            if not name:
                return None
            star = _i(u.get("star", 1), 1, 1, 3)
            items = [_s(x) for x in (u.get("items") or []) if x]
            pos_raw = u.get("position")
            pos = None
            if isinstance(pos_raw, (list, tuple)) and len(pos_raw) == 2:
                pos = (_i(pos_raw[0], 0), _i(pos_raw[1], 0))
            return Unit(name=name, star=star, items=items, position=pos)
        except Exception:
            return None

    def _coerce_trait(t: dict) -> Optional[ActiveTrait]:
        try:
            name = _s(t.get("name"))
            cnt = _i(t.get("count", 1), 1, 1, 20)
            tier = t.get("tier", "none")
            if tier not in {"bronze", "silver", "gold", "prismatic", "none"}:
                tier = "none"
            if not name:
                return None
            return ActiveTrait(name=name, count=cnt, tier=tier)
        except Exception:
            return None

    def _coerce_opp(o: dict) -> Optional[OpponentPreview]:
        try:
            return OpponentPreview(
                hp=_i(o.get("hp", 0), 0, 0, 100),
                top_carry=o.get("top_carry") or None,
                comp_summary=o.get("comp_summary") or None,
            )
        except Exception:
            return None

    def _coerce_bag(b: dict) -> Optional[BagItem]:
        try:
            return BagItem(slot=_i(b.get("slot", 0), 0, 0, 9), name=_s(b.get("name")))
        except Exception:
            return None

    return WorldState(
        stage=stage,
        round=_s(payload.get("round", "?-?")),
        hp=_i(payload.get("hp", 0), 0, 0, 100),
        gold=_i(payload.get("gold", 0), 0, 0, 999),
        level=_i(payload.get("level", 1), 1, 1, 10),
        exp=_s(payload.get("exp", "0/0")),
        board=[u for u in (_coerce_unit(x) for x in (payload.get("board") or [])) if u],
        bench=[u for u in (_coerce_unit(x) for x in (payload.get("bench") or [])) if u],
        bag=[b for b in (_coerce_bag(x) for x in (payload.get("bag") or [])) if b],
        shop=[_s(x) for x in (payload.get("shop") or []) if x],
        active_traits=[t for t in (_coerce_trait(x) for x in (payload.get("active_traits") or [])) if t],
        augments=[_s(x) for x in (payload.get("augments") or []) if x],
        opponents_preview=[o for o in (_coerce_opp(x) for x in (payload.get("opponents_preview") or [])) if o],
        timestamp=time.time(),
    )


class VLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        api_key: str = "EMPTY",
        timeout: float = 30.0,
        mode: str = "real",  # "real" | "mock"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.mode = mode

    async def _call(self, image_bytes: bytes, prompt: str) -> dict:
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions",
                                  json=payload, headers=headers)
            r.raise_for_status()
            body = r.json()
        content = body["choices"][0]["message"]["content"].strip()
        # 容错 markdown 代码块
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.lstrip().startswith("json"):
                content = content.split("\n", 1)[1]
            content = content.rstrip("`").strip()
        return json.loads(content)

    async def parse(self, image_bytes: bytes) -> WorldState:
        if self.mode == "mock":
            return self._mock_parse()
        try:
            data = await asyncio.wait_for(
                self._call(image_bytes, VLM_PARSE_PROMPT), timeout=self.timeout,
            )
            return _coerce_world_state(data)
        except Exception as e:
            log.warning("VLM parse 失败 · 降级 mock: %s", e)
            return self._mock_parse()

    def _mock_parse(self) -> WorldState:
        return WorldState(
            stage="unknown",
            round="?-?",
            hp=0, gold=0, level=1, exp="0/0",
            timestamp=time.time(),
        )
