"""S16 金铲铲版本知识提供器 —— 包一层 jcc-daida · 给 LLM 作 RAG。

jcc-daida 是外部项目 · 提供 Community Dragon + jcczz 混合数据源的 S16 (Set 16
"英雄联盟传奇") 结构化表 · 含英雄 / 羁绊 / 装备 / 海克斯 / meta 阵容。

加载路径优先级:
  1. 环境变量 JCC_DAIDA_PATH
  2. 默认本地开发路径 /mnt/c/Users/huannan/Downloads/带走/jcc-daida
  3. 不存在 → load_s16_knowledge() 返回 None · 由调用方 fallback

jcc-daida 没有 setup.py · 只能按文件路径加载。这里用 importlib 显式加载
client.py · 避免 `from client import ...` 污染全局命名空间。
"""
from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_DAIDA_PATH = Path("/mnt/c/Users/huannan/Downloads/带走/jcc-daida")


def _resolve_daida_path() -> Optional[Path]:
    env = os.environ.get("JCC_DAIDA_PATH")
    if env:
        p = Path(env).expanduser()
        return p if (p / "client.py").exists() else None
    if (_DEFAULT_DAIDA_PATH / "client.py").exists():
        return _DEFAULT_DAIDA_PATH
    return None


def _load_jcc_client_class(daida_path: Path):
    spec = importlib.util.spec_from_file_location(
        "_jcc_daida_client", daida_path / "client.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {daida_path}/client.py 加载 spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JCCClient


@dataclass
class Comp:
    name: str
    tier: str
    core_units: list[str]
    core_items: dict[str, list[str]]
    transitions: list[str] = field(default_factory=list)
    score: int = 0
    carry: Optional[str] = None
    play_style: Optional[str] = None


@dataclass
class S16Knowledge:
    comps: list[Comp]
    all_units: set[str]
    all_traits: set[str]
    all_items: set[str]
    all_augments: set[str]
    season_label: str = "英雄联盟传奇 (Set 16)"

    def version_context(self) -> str:
        top_carries: list[str] = []
        seen: set[str] = set()
        for c in self.comps:
            if c.carry and c.carry not in seen:
                top_carries.append(c.carry)
                seen.add(c.carry)
            if len(top_carries) >= 5:
                break
        carry_line = " / ".join(top_carries) if top_carries else "(无)"
        return (
            f"【当前版本 · S16 {self.season_label}】"
            f"共 {len(self.all_units)} 名英雄 · {len(self.all_traits)} 个羁绊 · "
            f"{len(self.all_items)} 件装备 · {len(self.all_augments)} 个海克斯强化。"
            f"版本主题为英雄联盟经典英雄联动 · 环境中有 {len(self.comps)} 套主流阵容。"
            f"高出场主 C(按综合评分):{carry_line}。"
            "复盘时请只用上述版本内的合法英雄/羁绊/装备名 · 不要引用旧版本内容。"
        )

    def comps_table(self) -> str:
        lines = [
            "| # | 阵容 | 分级 | 评分 | 主 C | 打法 | 核心单位 | 核心装备 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for idx, c in enumerate(self.comps, start=1):
            units = " / ".join(c.core_units[:8])
            item_bits = [
                f"{u}:{'、'.join(items)}"
                for u, items in c.core_items.items()
                if items
            ]
            items_col = " ; ".join(item_bits[:3]) if item_bits else "-"
            lines.append(
                f"| {idx} | {c.name} | {c.tier} | {c.score} | "
                f"{c.carry or '-'} | {c.play_style or '-'} | "
                f"{units} | {items_col} |"
            )
        return "\n".join(lines)

    def validate_unit_name(self, name: str) -> bool:
        return name in self.all_units


def _score_to_tier(score: int) -> str:
    if score >= 85:
        return "S"
    if score >= 80:
        return "A"
    if score >= 75:
        return "B"
    return "C"


def _extract_transitions(strategy: Optional[list]) -> list[str]:
    if not strategy:
        return []
    keywords = ("过渡", "前期", "中期", "后期", "拉", "D牌", "速")
    picked = [
        s.strip()
        for s in strategy
        if isinstance(s, str) and s.strip() and any(k in s for k in keywords)
    ]
    if picked:
        return picked[:4]
    return [s.strip() for s in strategy if isinstance(s, str) and s.strip()][:3]


def _build_comps(raw_comps: list, api_to_cn: dict[str, str], top_n: int) -> list[Comp]:
    accepted_sources = {"online_meta", "meta_seed", "community"}
    metas = [c for c in raw_comps if c.get("source") in accepted_sources]
    metas.sort(key=lambda c: -(c.get("score") or 0))
    out: list[Comp] = []
    for raw in metas[:top_n]:
        core_items: dict[str, list[str]] = {}
        for api_name, info in (raw.get("recommended_items") or {}).items():
            names = (info or {}).get("names") or []
            if not names:
                continue
            cn = api_to_cn.get(api_name, api_name)
            core_items[cn] = list(names)
        out.append(
            Comp(
                name=raw.get("name") or "(未命名)",
                tier=_score_to_tier(raw.get("score") or 0),
                core_units=list(raw.get("unit_names") or []),
                core_items=core_items,
                transitions=_extract_transitions(raw.get("strategy")),
                score=int(raw.get("score") or 0),
                carry=api_to_cn.get(raw.get("carry") or "", raw.get("carry")),
                play_style=raw.get("play_style"),
            )
        )
    return out


def load_s16_knowledge(top_n: int = 10) -> Optional[S16Knowledge]:
    """从 jcc-daida 加载 S16 版本知识 · 失败静默返回 None。"""
    daida_path = _resolve_daida_path()
    if daida_path is None:
        log.warning(
            "jcc-daida 路径未找到(JCC_DAIDA_PATH 环境变量未设置且默认路径不存在)· "
            "S16 知识库为空 · LLM 降级到通用 TFT 规则"
        )
        return None
    try:
        JCCClient = _load_jcc_client_class(daida_path)
        client = JCCClient(season="s16")
    except Exception as e:
        log.warning("加载 jcc-daida 失败 · %s · 降级", e)
        return None

    try:
        heroes = client._heroes
        traits = client._traits
        items = client._items
        augments = client._augments
        raw_comps = client._comps
        health = client.health()
    except Exception as e:
        log.warning("读取 jcc-daida 数据失败 · %s · 降级", e)
        return None

    api_to_cn = {h["api_name"]: h["name"] for h in heroes if h.get("api_name") and h.get("name")}
    comps = _build_comps(raw_comps, api_to_cn, top_n)

    return S16Knowledge(
        comps=comps,
        all_units={h["name"] for h in heroes if h.get("name")},
        all_traits={t["name"] for t in traits if t.get("name")},
        all_items={i["name"] for i in items if i.get("name")},
        all_augments={a["name"] for a in augments if a.get("name")},
        season_label=health.get("season_label", "英雄联盟传奇 (Set 16)"),
    )
