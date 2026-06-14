"""TFT version-knowledge provider — wraps jcc-daida · serves as RAG for the LLM.

jcc-daida is an external project · it provides a mixed data source of Community Dragon +
lolchess real top4_rate. Currently supports S16 (Academy / China-server live) and S17
(Starforged / PBE · current default) — the data shape differs slightly between them ·
S17 comps have no human-readable name but carry real top4_rate stats · this module
normalizes both into a Comp structure.

Load-path priority:
  1. environment variable JCC_DAIDA_PATH
  2. default local dev path /mnt/c/Users/you/Downloads/jcc-daida
  3. neither exists -> load_knowledge() returns None · the caller falls back

jcc-daida has no setup.py · it can only be loaded by file path. Here we load client.py
explicitly via importlib · to avoid `from client import ...` polluting the global namespace.

NOTE: the version_context() / comps_table() return strings, the _SEASON_LABELS values,
and the _extract_transitions keywords below are all Chinese on purpose — the first two are
fed to the LLM as RAG, the last is matched against Chinese strategy text.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_DAIDA_PATH = Path("/mnt/c/Users/you/Downloads/jcc-daida")


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
        raise ImportError(f"could not load spec from {daida_path}/client.py")
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
class Knowledge:
    comps: list[Comp]
    all_units: set[str]
    all_traits: set[str]
    all_items: set[str]
    all_augments: set[str]
    season: str = "s17"
    season_label: str = "星神 (Set 17 · PBE)"

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
            f"【当前版本 · {self.season.upper()} {self.season_label}】"
            f"共 {len(self.all_units)} 名英雄 · {len(self.all_traits)} 个羁绊 · "
            f"{len(self.all_items)} 件装备 · {len(self.all_augments)} 个海克斯强化。"
            f"环境中有 {len(self.comps)} 套主流阵容。"
            f"高出场主 C(按统计):{carry_line}。"
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


# Backward compatibility · old code that imports S16Knowledge still works
S16Knowledge = Knowledge


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
    # Chinese keywords matched against the (Chinese) strategy text — must stay Chinese.
    keywords = ("过渡", "前期", "中期", "后期", "拉", "D牌", "速")
    picked = [
        s.strip()
        for s in strategy
        if isinstance(s, str) and s.strip() and any(k in s for k in keywords)
    ]
    if picked:
        return picked[:4]
    return [s.strip() for s in strategy if isinstance(s, str) and s.strip()][:3]


def _comp_score(raw: dict) -> float:
    """Unified scoring · S17 has stats.top4_rate (real data) · S16 has score (0-86 rating)."""
    stats = raw.get("stats") or {}
    rate = stats.get("top4_rate")
    if isinstance(rate, (int, float)):
        return float(rate) * 100.0   # 0.86 → 86
    return float(raw.get("score") or 0)


def _build_comps(raw_comps: list, api_to_cn: dict[str, str], top_n: int) -> list[Comp]:
    # S16 uses online_meta / meta_seed / community · S17 uses meta · all accepted
    accepted_sources = {"online_meta", "meta_seed", "community", "meta"}
    metas = [c for c in raw_comps if c.get("source") in accepted_sources]
    metas.sort(key=lambda c: -_comp_score(c))
    out: list[Comp] = []
    for raw in metas[:top_n]:
        core_items: dict[str, list[str]] = {}
        # S16 has recommended_items · S17 is usually empty
        for api_name, info in (raw.get("recommended_items") or {}).items():
            names = (info or {}).get("names") or []
            if not names:
                continue
            cn = api_to_cn.get(api_name, api_name)
            core_items[cn] = list(names)

        carry_cn = api_to_cn.get(raw.get("carry") or "", raw.get("carry"))
        # S17 comps have no name · fall back to "<carry Chinese name> + Carry"
        # "(未命名)" (= unnamed) is rendered into the Chinese RAG comps table · kept Chinese.
        name = raw.get("name") or (f"{carry_cn} Carry" if carry_cn else "(未命名)")
        score = _comp_score(raw)

        out.append(
            Comp(
                name=name,
                tier=_score_to_tier(int(score)),
                core_units=list(raw.get("unit_names") or []),
                core_items=core_items,
                transitions=_extract_transitions(raw.get("strategy")),
                score=int(score),
                carry=carry_cn,
                play_style=raw.get("play_style"),
            )
        )
    return out


# Chinese season labels — rendered into version_context() RAG · kept Chinese.
_SEASON_LABELS = {
    "s17": "星神 (Set 17 · PBE)",
    "s16": "英雄联盟传奇 (Set 16)",
}


def load_knowledge(season: str = "s17", top_n: int = 10) -> Optional[Knowledge]:
    """Load the version knowledge for a given season from jcc-daida · silently returns None on failure.

    Defaults to s17 (the user's actual game season) · pass s16 to fall back to last season's learning.
    """
    daida_path = _resolve_daida_path()
    if daida_path is None:
        log.warning(
            "jcc-daida path not found (JCC_DAIDA_PATH is unset and the default path does not exist) · "
            "%s knowledge base is empty · LLM degrades to generic TFT rules", season.upper(),
        )
        return None
    try:
        JCCClient = _load_jcc_client_class(daida_path)
        client = JCCClient(season=season)
    except Exception as e:
        log.warning("failed to load jcc-daida (season=%s) · %s · degrading", season, e)
        return None

    try:
        heroes = client._heroes
        traits = client._traits
        items = client._items
        augments = client._augments
        raw_comps = client._comps
        health = client.health()
    except Exception as e:
        log.warning("failed to read jcc-daida data · %s · degrading", e)
        return None

    api_to_cn = {h["api_name"]: h["name"] for h in heroes if h.get("api_name") and h.get("name")}
    comps = _build_comps(raw_comps, api_to_cn, top_n)

    default_label = _SEASON_LABELS.get(season, season.upper())
    return Knowledge(
        comps=comps,
        all_units={h["name"] for h in heroes if h.get("name")},
        all_traits={t["name"] for t in traits if t.get("name")},
        all_items={i["name"] for i in items if i.get("name")},
        all_augments={a["name"] for a in augments if a.get("name")},
        season=season,
        season_label=health.get("season_label", default_label),
    )


def load_s16_knowledge(top_n: int = 10) -> Optional[Knowledge]:
    """Backward compatibility · explicitly load the S16 knowledge base. Equivalent to load_knowledge(season='s16')."""
    return load_knowledge(season="s16", top_n=top_n)
