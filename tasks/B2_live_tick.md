# B2 · real-time tick loop + replay collection

**Assigned to**: Claude Opus 4.7 (`claude-opus-4-7`) · core architecture · coordinates four modules (B1 data source / perception layer / B3 decision / B4 broadcast).
**Dependencies**: B1 (frame stream) · B3 (decision LLM) · B4 (broadcast service) **must be done first**.
**Estimated effort**: 1 day.
**Runtime**: **native Windows Python** (needs access to the OBS Virtual Camera).
**Role in the new product positioning**: **the central hub of the real-time coach**. Also the new entry point for the replay route (auto-calls `analyzer._llm_synthesize` at match end).

---

## Who you are

You are the Claude Opus 4.7 dispatched to `HuanNan520/jcc-replay-analyst` to execute B2.
B1/B3/B4 are already merged (you'll confirm before starting) · your task is to wire them into a closed-loop **real-time coach**.

At the same time · you wire up the "match end → auto-compose replay" side branch — reusing the existing `src/llm_analyzer.py` · without writing new LLM code.

## Data flow

```
OBSCapture.frames()                           ← B1
      ↓ bytes (PNG)
FrameMonitor.observe()                        ← existing
      ↓ keyframe trigger
VLMClient.parse(bytes)                        ← existing
      ↓ WorldState
_infer_decision_context(ws)                   ← you write this
      ↓ DecisionContext | None
                      ↓ None → only store in ring_buffer · don't call the LLM
                      ↓ present → call B3
DecisionLLM.decide(ws, ctx)                   ← B3
      ↓ Advice
POST /advice to B4                            ← B4
      ↓ also goes into ring_buffer

[match end · stage == "end"]
      ↓
Analyzer._llm_synthesize(ring_buffer)         ← existing · the whole-match WorldState sequence is fed in
      ↓
reports/S16-YYYYMMDD-HHMM.md + .json
```

## Target deliverable

```bash
# one command to start the real-time coach
python -m src.live_tick \
  --advice-server http://localhost:8765 \
  --llm-url http://localhost:8000/v1 \
  --llm-model Qwen3-VL-4B-FP8 \
  --fps 2 \
  --reports-dir reports/
```

---

## What to do

### 1. Add `src/live_tick.py`

```python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from .capture_obs import OBSCapture
from .frame_monitor import FrameMonitor, classify
from .vlm_client import VLMClient
from .ocr_client import find_number_near
from .schema import WorldState, MatchReport
from .decision_llm import DecisionLLM, DecisionContext, Advice
from .knowledge import load_knowledge
from .llm_analyzer import LocalLLMAnalyzer

log = logging.getLogger(__name__)


# ==================== Decision Context Inference ====================

def _infer_decision_context(ws: WorldState, prev_ws: Optional[WorldState]) -> Optional[DecisionContext]:
    """Infers from WorldState changes whether we've reached a decision point · returns a DecisionContext or None.

    Decision-point rules (conservative · prefer to miss rather than misfire):
    - augment: stage == 'augment' and the previous frame was not augment (newly popped) · options come from the last three of ws.augments or empty placeholders
    - carousel: stage == 'carousel' · options come from the 5 champion names in ws.shop
    - positioning: stage == 'positioning' · single trigger
    - level: stage just switched into 'pve' / 'pvp' · and gold >= (level+1)*4 (enough to level)
    - shop: skip · the shop is up every round · triggers too often · add it once user need is confirmed
    - item: bag grows to >= 2 and a new component appears (change detection)
    """
    if ws.stage == "augment" and (prev_ws is None or prev_ws.stage != "augment"):
        return DecisionContext(kind="augment", options=ws.augments[-3:] if ws.augments else ["?", "?", "?"])
    if ws.stage == "carousel" and (prev_ws is None or prev_ws.stage != "carousel"):
        return DecisionContext(kind="carousel", options=list(ws.shop[:5]))
    if ws.stage == "positioning" and (prev_ws is None or prev_ws.stage != "positioning"):
        return DecisionContext(kind="positioning", options=[])
    # level: at the start of a combat round · and economy clears the level-up threshold
    level_thresholds = {1: 0, 2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 96}
    if ws.stage in ("pve", "pvp") and (prev_ws is None or prev_ws.stage not in ("pve", "pvp")):
        required = level_thresholds.get(ws.level + 1, 999)
        if ws.gold >= required and ws.level < 9:
            return DecisionContext(kind="level", options=[])
    # item: bag grew
    if prev_ws is not None and len(ws.bag) >= 2 and len(ws.bag) > len(prev_ws.bag):
        return DecisionContext(kind="item", options=[b.name for b in ws.bag])
    return None


# ==================== Advice Publisher ====================

class AdvicePublisher:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=3.0)
        return self

    async def __aexit__(self, *a):
        await self._client.aclose()

    async def publish(self, advice: Advice) -> None:
        try:
            r = await self._client.post(
                f"{self.server_url}/advice",
                json=advice.model_dump(),
            )
            r.raise_for_status()
            log.debug("advice published · %s · broadcast_to=%s", advice.kind, r.json().get("broadcast_to"))
        except Exception as e:
            log.warning("advice publish failed · %s · %s", advice.kind, e)


# ==================== Match Report Saver ====================

def _save_report(report: MatchReport, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md_path = reports_dir / f"{report.match_id}-{stamp}.md"
    json_path = md_path.with_suffix(".json")

    # reuses scripts/analyze.py:report_to_markdown · but that's a CLI-internal function · just clone it here
    # NOTE: the report body below is the generated Chinese match report · kept as a functional output template
    lines = [
        f"# 对局复盘 · {report.match_id}",
        "",
        f"- 最终排名：**{report.final_rank}** / 8",
        f"- 最终血量：{report.final_hp}",
        f"- 对局时长：{report.duration_s} 秒",
        f"- 核心阵容：{report.core_comp or '未识别'}",
        "",
        "## 关键回合",
        "",
    ]
    for r in report.key_rounds:
        lines += [
            f"### {r.round} · {r.title}",
            f"**评级：{r.grade}**" + (f"　{r.delta}" if r.delta else ""),
            "",
            r.comment,
            "",
        ]
    lines += ["## AI 总评", "", report.summary, ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info("replay saved · %s", md_path)
    return md_path


# ==================== Live Tick Loop ====================

class LiveTickLoop:
    def __init__(
        self,
        capture: OBSCapture,
        vlm: VLMClient,
        decision_llm: DecisionLLM,
        post_match_llm: LocalLLMAnalyzer,
        publisher: AdvicePublisher,
        reports_dir: Path,
        ring_size: int = 60,
    ):
        self.capture = capture
        self.monitor = FrameMonitor(screen_size=(2560, 1456))
        self.vlm = vlm
        self.decision_llm = decision_llm
        self.post_match_llm = post_match_llm
        self.publisher = publisher
        self.reports_dir = reports_dir
        self.ring: deque[WorldState] = deque(maxlen=ring_size)
        self._prev_ws: Optional[WorldState] = None
        self._match_started = False

    async def run(self) -> None:
        log.info("LiveTickLoop started · reports_dir=%s", self.reports_dir)
        async for frame in self.capture.frames():
            events = self.monitor.observe(frame)
            if not self.monitor.any_triggered(events) and self._match_started:
                continue   # no keyframe change · save the VLM call
            event_kind = classify(self.monitor.changed_regions(events))
            log.debug("frame event=%s", event_kind)

            try:
                ws = await self.vlm.parse(frame)
            except Exception as e:
                log.warning("VLM parse failed · %s", e)
                continue

            # filter out invalid states (the perception layer may return unknown)
            if ws.stage == "unknown" and self._prev_ws is None:
                continue

            # first valid stage seen · mark the match as started
            if ws.stage != "unknown" and not self._match_started:
                self._match_started = True
                log.info("match started · round=%s stage=%s", ws.round, ws.stage)

            self.ring.append(ws)

            # match end · trigger replay
            if ws.stage == "end" and self._prev_ws is not None and self._prev_ws.stage != "end":
                await self._finalize_match()
                self._prev_ws = ws
                continue

            # decision-point check
            ctx = _infer_decision_context(ws, self._prev_ws)
            if ctx:
                log.info("decision point triggered · kind=%s round=%s", ctx.kind, ws.round)
                try:
                    advice = await self.decision_llm.decide(ws, ctx)
                    await self.publisher.publish(advice)
                except Exception as e:
                    log.warning("decision chain failed · %s", e)

            self._prev_ws = ws

    async def _finalize_match(self) -> None:
        log.info("match ended · composing replay · ring size=%d", len(self.ring))
        try:
            report = await self.post_match_llm.synthesize(list(self.ring))
            _save_report(report, self.reports_dir)
        except Exception as e:
            log.error("replay synthesis failed · %s", e)
        self.ring.clear()
        self._match_started = False
        self._prev_ws = None


# ==================== CLI ====================

async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice-server", default="http://localhost:8765")
    ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
    ap.add_argument("--vlm-model", default="Qwen3-VL-4B-FP8")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="Qwen3-VL-4B-FP8")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    knowledge = load_knowledge()   # default season='s17' · the live season
    capture = OBSCapture(fps=args.fps)
    vlm = VLMClient(base_url=args.vlm_url, model=args.vlm_model, mode="real")
    decision_llm = DecisionLLM(base_url=args.llm_url, model=args.llm_model, knowledge=knowledge)
    post_match = LocalLLMAnalyzer(base_url=args.llm_url, model=args.llm_model, knowledge=knowledge)

    async with AdvicePublisher(args.advice_server) as publisher:
        loop = LiveTickLoop(
            capture=capture,
            vlm=vlm,
            decision_llm=decision_llm,
            post_match_llm=post_match,
            publisher=publisher,
            reports_dir=args.reports_dir,
        )
        await loop.run()


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
```

### 2. Unit test `tests/test_live_tick.py`

Focus on the decision logic in `_infer_decision_context` (no real LLM / real OBS needed):

```python
import time
import pytest
from src.live_tick import _infer_decision_context
from src.schema import WorldState


def _ws(stage, round="1-1", hp=100, gold=0, level=1, bag=None):
    return WorldState(
        stage=stage, round=round, hp=hp, gold=gold, level=level,
        exp="0/0", timestamp=time.time(),
        bag=bag or [],
    )


class TestDecisionContextInference:
    def test_augment_first_entry(self):
        ctx = _infer_decision_context(_ws("augment"), prev_ws=None)
        assert ctx and ctx.kind == "augment"

    def test_augment_not_retriggered_if_still_augment(self):
        prev = _ws("augment")
        ctx = _infer_decision_context(_ws("augment"), prev_ws=prev)
        assert ctx is None

    def test_carousel_first_entry(self):
        ctx = _infer_decision_context(_ws("carousel"), prev_ws=None)
        assert ctx and ctx.kind == "carousel"

    def test_positioning_first_entry(self):
        ctx = _infer_decision_context(_ws("positioning"), prev_ws=_ws("pvp"))
        assert ctx and ctx.kind == "positioning"

    def test_level_triggered_when_gold_enough(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=10, level=3)  # leveling to 4 needs 10 gold
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx and ctx.kind == "level"

    def test_level_not_triggered_when_insufficient_gold(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=4, level=3)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_level_capped_at_9(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=500, level=9)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_pve_to_pvp_not_retriggered(self):
        prev = _ws("pve", gold=10, level=3)
        curr = _ws("pvp", gold=8, level=4)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None   # already in a combat round · don't re-trigger level

    def test_no_trigger_on_pvp_steady(self):
        prev = _ws("pvp")
        curr = _ws("pvp")
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None
```

### 3. Minor tweak: the interface `Analyzer` exposes and the `LocalLLMAnalyzer` reuse

Look at `src/analyzer.py` · if `_llm_synthesize` is a method of `Analyzer` that actually calls `LocalLLMAnalyzer.synthesize` — then instantiating `LocalLLMAnalyzer(...)` directly works the same · the code above already uses it this way.

**Do not modify `analyzer.py`** — keep the replay path's CLI entry usable.

---

## What not to do

- Do not write your own LLM calls · use B3's DecisionLLM + the existing LocalLLMAnalyzer directly
- Do not modify any of `src/schema.py` · `src/vlm_client.py` · `src/llm_analyzer.py` · `src/knowledge.py`
- Do not write a "VLM every frame" brute-force version — FrameMonitor's keyframe triggering is the core compute saver
- Do not introduce queue.Queue / threading · all asyncio
- Do not hardcode screen_size · use FrameMonitor's auto orientation inference (or have B1 return the resolution)
- Do not write daemon / supervisor logic — if it crashes, it crashes · let a shell systemd / nssm above guard it
- Do not persist the ring buffer to disk · fully in-memory · drop a report at match end
- Do not write a special branch for every event kind — `classify`'s return value is currently only used for debug logging

---

## Self-acceptance checklist

- [ ] `python -c "from src.live_tick import LiveTickLoop, _infer_decision_context, AdvicePublisher, _save_report"` imports without error
- [ ] `pytest tests/test_live_tick.py -v` all green · at least 9 tests
- [ ] `python -m src.live_tick --help` prints the full argument list
- [ ] **Integration test · requires B1/B3/B4 all running**:
  - Start vLLM (8000) + advice_server (8765) + the OBS Virtual Camera (with MuMu open)
  - Run `python -m src.live_tick --fps 1`
  - Enter the game carousel or augment pick · check whether advice arrives at `ws://localhost:8765/ws/advice` (verify with websocat)
  - The advice JSON is valid · contains kind/reasoning/confidence
- [ ] Play through to the end stage · `.md` and `.json` files appear in the `reports/` directory
- [ ] Run the original 40 pytest + the 9 new ones · zero regressions
- [ ] `git diff --stat` only contains:
  - `src/live_tick.py` (new)
  - `tests/test_live_tick.py` (new)
  - `README.md` (optional · add a small run section)

## After completion

Give the user a ≤ 200-word report:
- Over one full match · how many of each decision kind triggered (augment × N / positioning × N / ...)
- Average per-frame processing time (FrameMonitor + VLM + possible LLM)
- At match end · the replay file path + size
- Observed **false negatives / false positives** of `_infer_decision_context` (e.g. missed a key node / triggered one extra)
- TODO leads for later tuning

No git commit.

---

## References

- FrameMonitor.classify's event types · see `src/frame_monitor.py:216`
- A1's `src/llm_analyzer.py` is reused directly as the post-match analyzer · don't change it
- B3's `DecisionLLM` / `DecisionContext` / `Advice` · see `src/decision_llm.py`
- B4's `/advice` endpoint body shape · see `src/advice_server.py`
