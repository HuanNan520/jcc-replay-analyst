"""B3 smoke demo · run one augment decision against a real vLLM · observe latency and output.

Usage:
    # start vLLM first (shares the instance with A1 · port 8000)
    python scripts/demo_decision.py

Requires a local vLLM serving an OpenAI-compatible API at http://localhost:8000 · model Qwen3-VL-4B-FP8.

Knowledge loading strategy:
  1. try S17 first · on failure / empty data, fall back to S16 + a TODO note
  2. if both fail, knowledge=None · the LLM falls back to generic TFT rules
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# allow `python scripts/demo_decision.py` to run directly from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.decision_llm import DecisionContext, DecisionLLM  # noqa: E402
from src.knowledge import load_s16_knowledge  # noqa: E402
from src.schema import ActiveTrait, BagItem, Unit, WorldState  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s · %(message)s",
)
log = logging.getLogger("demo_decision")


def _try_load_s17_knowledge():
    """Try to load S17 knowledge · returns the object on success · returns (None, reason) on failure.

    jcc-daida S17 currently has 73 heroes / 44 traits / 103 comps · but the `source` tag is
    `meta` / `variant` (S16 uses `online_meta` / `meta_seed` / `community`) ·
    the existing `load_s16_knowledge` filter does not match · so even if it loads it returns 0 comps.

    TODO(A3): extend `load_s16_knowledge(season=...)` to support the S17 comp source ·
    or make `_build_comps`'s `accepted_sources` configurable. For now use a manual adapter.
    """
    daida_path = os.environ.get("JCC_DAIDA_PATH", "/mnt/c/Users/you/Downloads/jcc-daida")
    if not (Path(daida_path) / "client.py").exists():
        return None, f"jcc-daida path does not exist {daida_path}"

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_jcc_daida_client_s17", Path(daida_path) / "client.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = mod.JCCClient(season="s17")
        health = client.health()
        log.info("jcc-daida S17 health: %s", health)
        heroes = client._heroes
        traits = client._traits
        items = client._items
        augments = client._augments
        raw_comps = client._comps
    except Exception as e:
        return None, f"failed to load S17 · {type(e).__name__}: {e}"

    if not heroes:
        return None, "S17 heroes is empty"

    # manually adapt S17 comps · source is meta/variant · no score
    api_to_cn = {
        h["api_name"]: h["name"]
        for h in heroes
        if h.get("api_name") and h.get("name")
    }

    from src.knowledge import Comp, S16Knowledge

    metas = [c for c in raw_comps if c.get("source") == "meta"]
    # ascending by stats.place (smaller number = higher placement)
    metas.sort(key=lambda c: (c.get("stats") or {}).get("place", 9.0))

    comps: list[Comp] = []
    for idx, raw in enumerate(metas[:10], start=1):
        stats = raw.get("stats") or {}
        place = stats.get("place", 5.0)
        # place below 4.0 counts as strong · map to S/A/B tier
        if place <= 3.5:
            tier = "S"
        elif place <= 4.0:
            tier = "A"
        elif place <= 4.5:
            tier = "B"
        else:
            tier = "C"
        carry_api = raw.get("carry") or ""
        carry_cn = api_to_cn.get(carry_api, carry_api) if carry_api else None
        comps.append(
            Comp(
                name=f"S17-{idx} · carry {carry_cn or '?'}",
                tier=tier,
                core_units=list(raw.get("unit_names") or []),
                core_items={},
                transitions=[],
                score=int(max(0, min(100, (5.0 - place) * 30))),
                carry=carry_cn,
                play_style=None,
            )
        )

    return S16Knowledge(
        comps=comps,
        all_units={h["name"] for h in heroes if h.get("name")},
        all_traits={t["name"] for t in traits if t.get("name")},
        all_items={i["name"] for i in items if i.get("name")},
        all_augments={a["name"] for a in augments if a.get("name")},
        season_label=health.get("season_label", "星神 / Space Gods (Set 17)"),
    ), None


async def main(base_url: str, model: str, kind: str) -> int:
    # knowledge: S17 -> S16 -> None
    k, reason = _try_load_s17_knowledge()
    if k is None:
        log.warning("S17 knowledge failed to load · falling back to S16 · reason=%s", reason)
        k = load_s16_knowledge()
        if k is None:
            log.warning("S16 knowledge also failed · knowledge=None · LLM falls back to generic rules")
    else:
        log.info(
            "S17 knowledge ready · %d heroes · %d comps · label=%s",
            len(k.all_units), len(k.comps), k.season_label,
        )

    ws = WorldState(
        stage="augment",
        round="2-1",
        hp=82,
        gold=28,
        level=5,
        exp="12/20",
        board=[
            Unit(name="安妮", star=2),
            Unit(name="阿狸", star=1),
        ],
        bench=[Unit(name="剑圣", star=1)],
        bag=[
            BagItem(slot=0, name="暴风大剑"),
            BagItem(slot=1, name="反曲之弓"),
        ],
        shop=["亚索", "卢锡安", "寒冰", "瑟庄妮", "李青"],
        active_traits=[ActiveTrait(name="法师", count=2, tier="bronze")],
        augments=[],
        timestamp=time.time(),
    )

    ctx = DecisionContext(
        kind=kind,  # type: ignore[arg-type]
        options=["法师之力", "复利", "攻速强化"],
        timeout_s=25,
    )

    llm = DecisionLLM(base_url=base_url, model=model, knowledge=k, timeout=20.0)

    print(f"\n=== DecisionLLM · kind={kind} ===")
    t0 = time.time()
    advice = await llm.decide(ws, ctx)
    dt = time.time() - t0
    print(f"elapsed: {dt:.2f}s")
    print(advice.model_dump_json(indent=2))

    if advice.confidence == 0.0 and "降级" in advice.reasoning:
        print("\n[WARN] fallback triggered · LLM unavailable or output invalid")
        return 2
    if dt > 3.0:
        print(f"\n[WARN] elapsed {dt:.2f}s exceeds the 3s budget")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3-VL-4B-FP8")
    p.add_argument(
        "--kind",
        default="augment",
        choices=["augment", "carousel", "shop", "level", "positioning", "item"],
    )
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.base_url, args.model, args.kind)))
