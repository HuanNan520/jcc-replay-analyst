"""B3 smoke demo · 对真 vLLM 跑一次 augment 决策 · 看延迟和输出。

使用方式：
    # 先把 vLLM 跑起来（和 A1 共享实例 · 端口 8000）
    python scripts/demo_decision.py

需要本地 vLLM 在 http://localhost:8000 上提供 OpenAI 兼容接口 · 模型 Qwen3-VL-4B-FP8。

Knowledge 加载策略：
  1. 先尝试 S17 · 失败 / 空数据则回退 S16 + TODO 标注
  2. 都失败就 knowledge=None · LLM 降级到通用 TFT 规则
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# 允许 `python scripts/demo_decision.py` 直接从 repo 根运行
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
    """尝试加载 S17 knowledge · 成功返回对象 · 失败返回 (None, reason)。

    jcc-daida S17 目前有 73 英雄 / 44 羁绊 / 103 comps · 但 `source` 标签是
    `meta` / `variant`（S16 走 `online_meta` / `meta_seed` / `community`）·
    现有 `load_s16_knowledge` 的过滤器不匹配 · 即使能 load 也会返回 0 comps。

    TODO(A3): 扩展 `load_s16_knowledge(season=...)` 支持 S17 comp source ·
    或者让 `_build_comps` 的 `accepted_sources` 可配。现在先用手工 adapter。
    """
    daida_path = os.environ.get("JCC_DAIDA_PATH", "/mnt/c/Users/huannan/Downloads/带走/jcc-daida")
    if not (Path(daida_path) / "client.py").exists():
        return None, f"jcc-daida 路径不存在 {daida_path}"

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
        return None, f"加载 S17 失败 · {type(e).__name__}: {e}"

    if not heroes:
        return None, "S17 heroes 为空"

    # 手工适配 S17 comp · source 是 meta/variant · 没有 score
    api_to_cn = {
        h["api_name"]: h["name"]
        for h in heroes
        if h.get("api_name") and h.get("name")
    }

    from src.knowledge import Comp, S16Knowledge

    metas = [c for c in raw_comps if c.get("source") == "meta"]
    # 按 stats.place 升序（数字越小名次越靠前）
    metas.sort(key=lambda c: (c.get("stats") or {}).get("place", 9.0))

    comps: list[Comp] = []
    for idx, raw in enumerate(metas[:10], start=1):
        stats = raw.get("stats") or {}
        place = stats.get("place", 5.0)
        # place 4.0 以下算强 · 转成 S/A/B tier
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
                name=f"S17-{idx} · 主 C {carry_cn or '?'}",
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
    # 知识库：S17 → S16 → None
    k, reason = _try_load_s17_knowledge()
    if k is None:
        log.warning("S17 知识库加载失败 · 回退 S16 · reason=%s", reason)
        k = load_s16_knowledge()
        if k is None:
            log.warning("S16 知识库也失败 · knowledge=None · LLM 降级到通用规则")
    else:
        log.info(
            "S17 知识库就绪 · %d heroes · %d comps · label=%s",
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
    print(f"耗时: {dt:.2f}s")
    print(advice.model_dump_json(indent=2))

    if advice.confidence == 0.0 and "降级" in advice.reasoning:
        print("\n[WARN] Fallback 被触发 · LLM 不可用或输出非法")
        return 2
    if dt > 3.0:
        print(f"\n[WARN] 耗时 {dt:.2f}s 超过 3s 预算")
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
