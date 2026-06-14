# jcc-replay-analyst · task allocation overview

---

## Done · A1–A4 (MVP · screen-recording replay)

| #  | Task                          | Status   | Key output |
|----|-------------------------------|----------|------------|
| A1 | Hook a local LLM into `_llm_synthesize` | ✅ done | `src/llm_analyzer.py` · vLLM + Qwen guided_json |
| A2 | Fill in examples/sample_frames/ | ✅ done | 12 real S16 frames |
| A3 | Hook jcc-daida as the S16 RAG | ✅ done | `src/knowledge.py` · 113 champions / 53 traits / 10 comps |
| A4 | Initial test suite + CI       | ✅ done | `tests/` × 40 passing · `.github/workflows/ci.yml` |

**MVP status**: clone and run the mock pipeline immediately · a local vLLM lets you run the real LLM · CI green.

---

## Product repositioning · from replay to real-time coach

**New positioning**: **real-time AI coach** (primary route) + **automatic replay** (byproduct)

While playing TFT · the OBS Virtual Camera streams the game window → Python pulls frames continuously → the perception layer recognizes → a decision point triggers a low-latency LLM → advice floats above the game window as a translucent card. When the match ends, a full-match replay is composed automatically and saved to `reports/`.

### Three red lines

1. **Read-only, no input** — sends no ADB input / keyboard-mouse simulation
2. **Local inference by default** — can be forked to the cloud · but zero API cost by default
3. **No Android side** — goes through the OBS Virtual Camera · not ADB debugging · Tencent anti-cheat cannot perceive this layer

---

## B1–B5 · real-time coach pipeline (all ✅ done)

| #  | Task                          | Executor AI         | Core output | Status |
|----|-------------------------------|---------------------|-------------|--------|
| B1 | OBS Virtual Camera data source | **Sonnet 4.6**     | `src/capture_obs.py` · native Windows Python | ✅ |
| B2 | real-time tick loop + replay collection | **Opus 4.7** | `src/live_tick.py` · coordinates perception → decision → broadcast | ✅ |
| B3 | low-latency decision LLM (6 kinds) | **Opus 4.7**    | `src/decision_llm.py` · 6 short prompts ≤3s | ✅ |
| B4 | WebSocket advice push service | **Sonnet 4.6**     | `src/advice_server.py` · FastAPI + `/ws/advice` | ✅ |
| B5 | PyQt desktop overlay          | **Opus 4.7**        | `src/overlay_ui.py` · Win32 FindWindow + translucent card | ✅ |

## C1–C14 · polish + portfolio-readiness (8 items done)

| #  | Task                          | Status | Output |
|----|-------------------------------|--------|--------|
| C1 | Hallucination audit hardening | ✅     | llm_analyzer tokenization scan + whitelist |
| C4 | Reports HTML renderer         | ✅     | `src/report_renderer.py` + demo_report.py + `examples/sample_report.html` |
| C9 | README architecture diagram rework | ✅ | ASCII dual-route + table LLM row changed to local vLLM |
| C10 | e2e smoke end-to-end demo    | ✅     | `scripts/e2e_smoke.py` · runs in WSL in 50s |
| C11 | pitch/index.html rework      | ✅     | 1451-line AI-coach landing |
| C12 | Positioning decode fix       | ✅     | pydantic field_validator coerce |
| C13 | Environment self-check script | ✅    | `scripts/setup_check.py` · 19 checks |
| C14 | Windows one-click launch     | ✅     | `scripts/start_coach.bat/.ps1` |

### Parallelization strategy

```
Phase 1 · three windows open at once
  B1 (Sonnet) ── frames() stream            ─┐
  B3 (Opus)   ── DecisionLLM + 6 schemas    ─┤
  B4 (Sonnet) ── WebSocket broadcast        ─┘

Phase 2 · B2 (Opus) ── assembles live_tick · consumes B1/B3/B4

Phase 3 · B5 (Opus) ── PyQt overlay · consumes B4's WS client
```

Only one Opus can run serially · so the three Opus segments go in order: **B3 → B2 → B5**. Sonnet's B1/B4 run in parallel while B3 is executing.

---

## Common constraints (all B tasks)

1. **The three red lines** above · each task self-verifies them up front
2. **YAGNI** — do nothing outside the task · no premature abstraction
3. **Reuse existing code** — perception layer / schema / knowledge change essentially zero · only add new files
4. **Native Windows Python code** — explicitly required for B1 / B5 · B4 ideally runs on both sides too
5. **No commits** — sub-agents only produce a diff · don't push · hand it back to me for review

---

## Repository entry

Already cloned at `~/jcc-replay-analyst` (WSL side) · A1-A4 outputs are already on the main branch.

The Windows side needs to be set up separately:

```powershell
# Windows PowerShell
cd C:\Users\you\Downloads
git clone https://github.com/HuanNan520/jcc-replay-analyst.git jcc-replay-analyst-win
cd jcc-replay-analyst-win
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt   # B1/B5 will create this
```

---

## Task files

- [`B1_obs_capture.md`](./B1_obs_capture.md) — OBS Virtual Camera data source
- [`B2_live_tick.md`](./B2_live_tick.md) — real-time tick loop + replay collection
- [`B3_decision_llm.md`](./B3_decision_llm.md) — low-latency decision LLM
- [`B4_advice_server.md`](./B4_advice_server.md) — WebSocket push
- [`B5_overlay_ui.md`](./B5_overlay_ui.md) — PyQt desktop overlay

## How to start · same as phase A

```bash
# three windows in parallel
cd ~/jcc-replay-analyst
source .venv/bin/activate
claude
/model sonnet     # B1/B4
/model opus       # B3/B2/B5
```

Paste this first:
```
Read the entire contents of tasks/BX_XXX.md · follow its constraints and steps exactly · run every self-acceptance checklist item green before reporting back to me · do not touch files outside its scope · do not git commit yourself.
```
