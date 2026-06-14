# jcc-replay-analyst

[![CI](https://github.com/HuanNan520/jcc-replay-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/HuanNan520/jcc-replay-analyst/actions/workflows/ci.yml)

🌐 **Live** · [pitch landing](https://huannan520.github.io/jcc-replay-analyst/) · [engineering roadmap](https://huannan520.github.io/jcc-replay-analyst/roadmap.html) · [build log](https://huannan520.github.io/jcc-replay-analyst/build_log.html) · [real-LLM sample replay](https://huannan520.github.io/jcc-replay-analyst/sample_report.html) (first time: repo Settings → Pages → Source → "GitHub Actions", then run Deploy Pages manually from the Actions tab)

> Teamfight Tactics (TFT, China-server mobile, jkchess) AI coach · **real-time in-game advice** + **automatic post-match replay analysis**, two entry points in one.

<p align="center">
  <em>Read-only screen capture · never touches the game · driven by the OBS Virtual Camera · compliant</em>
</p>

## The problem

Existing TFT tools only look up **composition tables** — they tell you "this comp is strong." Nobody tells you **while you play** **which augment to pick · whether to level up · who to give items to**, and nobody tells you afterward **which step in this match went wrong**.

This project fills both gaps:
- **Real-time coach**: while the player is playing · OBS pushes the MuMu emulator window out as a virtual camera · Python recognizes the screen continuously · decision points trigger the LLM · advice fades in as a translucent card above the MuMu window
- **Automatic replay analysis**: when the match ends, a full-match replay report is composed automatically into `reports/` (no manual screen recording needed)

The same perception layer (VLM + OCR + CV + S16 knowledge RAG) drives both routes.

## Sample report

📊 [`examples/sample_report.html`](./examples/sample_report.html) — **a real LLM-generated HTML replay** (open in a browser · dark visual · produced by `scripts/demo_report.py` running a local vLLM)

📄 [`examples/sample_report.md`](./examples/sample_report.md) — the older markdown mock (kept for comparison · handy for reading the field structure)

🎨 [`pitch/index.html`](./pitch/index.html) — full visual presentation (open in a browser)

🗺️ [`pitch/roadmap.html`](./pitch/roadmap.html) — engineering roadmap (A/B/C phase progress visualization)

## End-to-end smoke demo

No real OBS · no Windows required · runs the full pipeline inside WSL · produces an HTML replay in under 50 seconds:

```bash
python3 scripts/e2e_smoke.py
# When vLLM is available, really runs 5 decisionLLM kinds + full-match replay synthesis
# When vLLM is unavailable, falls back automatically · still runs the whole pipeline
# Output: /tmp/e2e_smoke_report.html (≈ 18 KB)
```

## Technical architecture

```
                    Data source layer (pick one)
   ┌──────────────────────┬──────────────────────┐
   │  OBS Virtual Camera  │  mp4 recording / frame seq │
   │  (real-time coach)   │  (post-match replay)  │
   └──────────┬───────────┴──────────┬───────────┘
              │                      │
              ▼                      ▼
      ┌───────────────────────────────────────┐
      │  Shared perception layer              │
      │  frame_monitor (dHash keyframes)       │
      │  ocr_client (PaddleOCR digits)         │
      │  vlm_client (Qwen3-VL semantics)       │
      │  → WorldState                         │
      └──────────┬────────────────┬───────────┘
                 │                │
                 ▼                ▼
     ┌─────────────────┐  ┌──────────────────┐
     │  Real-time coach│  │  Post-match replay│
     │  live_tick →    │  │  analyzer →       │
     │  decision_llm   │  │  llm_analyzer →   │
     │  (6 kinds ≤3s)→ │  │  MatchReport →    │
     │  advice_server  │  │  md/json/html     │
     │  → overlay_ui   │  │  (reuses the      │
     │  (translucent   │  │   real-time path's│
     │   card)         │  │   end-of-match    │
     │                 │  │   collection)     │
     └─────────────────┘  └──────────────────┘

            S17 patch knowledge base (jcc-daida) ·
            → KnowledgeProvider injected into both routes
```

**Why the hybrid architecture** · let specialized models do specialized work:

| Layer | What it does | Why not hand it to the VLM |
| --- | --- | --- |
| OCR · PaddleOCR | Reads HP / gold / level digits | Qwen2.5-VL often misreads HP by 1-2 · OCR is 99%+ stable |
| CV · OpenCV | Finds item icons · highlights UI elements | 30 lines of color/shape matching is enough · fast and accurate |
| VLM · Qwen2.5-VL | Trait semantics · composition classification | What a VLM should do · don't make it read digits |
| LLM · local vLLM | Generates scoring and narrative | Needs TFT-patch knowledge-base RAG · local by default, zero API cost |

**Each layer fails independently without contaminating the others** · far more robust than betting on a single large VLM.

## Module overview

```
src/
├── adb_client.py      ADB screencap (screencap / screenrecord fallback) · sends no input commands
├── frame_monitor.py   dHash perceptual hash · detects screen change by ROI · extracts keyframes
├── ocr_client.py      PaddleOCR wrapper · recognize / find_number_near / long-bubble detection
├── arrow_finder.py    OpenCV · finds the center of bright UI elements on screen
├── vlm_client.py      Qwen VLM · recognizes on-screen semantic fields → WorldState
├── schema.py          WorldState / RoundReview / MatchReport data structures
└── analyzer.py        pipeline · takes a frame sequence, emits a MatchReport
scripts/
└── analyze.py         CLI entry point
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the bare skeleton (mock VLM · mock LLM · verifies the pipeline)
python scripts/analyze.py --frames examples/sample_frames/ --vlm mock --llm mock --out report.md

# 3. Hook up a real VLM (needs a local vLLM serving Qwen2.5-VL or Qwen3-VL)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --port 8000 &
python scripts/analyze.py --frames path/to/frames/ --vlm real --out report.md

# 4. Extract frames straight from an mp4 recording and analyze
python scripts/analyze.py --video match.mp4 --every-s 5 --vlm real --out report.md

# 5. Hook up the local LLM analysis layer (analysis and perception share the same vLLM instance)
#    With ≥ 14GB free VRAM use Qwen3-VL-8B-FP8 · if tight (e.g. a 16GB card also running the emulator) drop to Qwen3-VL-4B-FP8
python scripts/analyze.py \
  --frames examples/sample_frames/ \
  --vlm mock --llm real \
  --llm-url http://localhost:8000/v1 \
  --llm-model Qwen3-VL-4B-FP8 \
  --out report.md
# You can also run VLM + LLM for real together · just swap --vlm mock for --vlm real
```

### Runtime · local inference by default

**Runs on local vLLM by default · zero API cost · works out of the box**. Not locked to local — if you want to pay for a cloud API (Claude / OpenAI / others), write a `CloudLLMAnalyzer` that satisfies the same interface and swap it in.

- The perception layer (VLM) runs on local vLLM · OpenAI-compatible interface
- The analysis layer (LLM) also runs on local vLLM · by default reuses the same instance and model as perception
- No `anthropic` / `openai` hard dependency by default (cloud API cost is far higher than self-hosting Qwen) · uses only `httpx` to hit `/v1/chat/completions` directly
- Structured output prefers vLLM's native `guided_json` · falls back to `response_format: json_object`
- Zero network dependency · runs locally out of the box — moving to the cloud only means writing another Analyzer implementation

### Real-time coach mode · data source

Real-time mode requires native Windows Python (not WSL) · prerequisites:

1. Install OBS Studio and launch it
2. Add a "Window Capture" to grab the MuMu window
3. Click "Start Virtual Camera" at the bottom right of OBS
4. `pip install -r requirements-windows.txt`
5. `python scripts/test_obs_capture.py` to verify the connection

### Real-time coach mode · full startup (4 terminals · after B2 is done)

```powershell
# Terminal 1 · WSL · start vLLM (shared by perception + analysis + decision)
source ~/jcc-replay-analyst/.venv/bin/activate
python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-VL-4B-FP8 --port 8000

# Terminal 2 · Windows or WSL · start the advice broadcast service
python -m src.advice_server --port 8765

# Terminal 3 · native Windows Python · start OBS virtual cam + live tick (requires OBS Start Virtual Camera first)
python -m src.live_tick --fps 2 --advice-server http://localhost:8765

# Terminal 4 · native Windows Python · start the overlay (translucent card floating above the MuMu window)
python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice
```

The overlay automatically finds the MuMu main window and docks to it · cards fade in when a decision point triggers · and fade out after 8 seconds.
For debugging, `--no-click-through` lets you click the overlay itself.

**Or one command** (Windows): `scripts\start_coach.bat` (or `.ps1`) · opens 4 terminals and runs the whole chain automatically.

## Project status

**198 pytest passing · CI green · 15 task commits** · three routes:

### A · Replay route (MVP done)
- ✅ Perception layer (frame_monitor · ocr · vlm) working · stable recognition on real TFT footage
- ✅ Pipeline skeleton · mock mode runs the whole flow and emits a markdown report
- ✅ LLM analysis layer hooked to local vLLM · produces real replay analysis with quantified counterfactuals (`src/llm_analyzer.py`)
- ✅ S17 patch knowledge base via `jcc-daida` · 73 champions / 37 traits / 591 items / 10 comps ranked by top4_rate (`src/knowledge.py`)
- ✅ 40 pytest green · GitHub Actions CI passing

### B · Real-time coach route (done · see `tasks/README.md`)
- ✅ B1 · OBS Virtual Camera data source (`src/capture_obs.py` · replaces ADB · more compliant and more stable)
- ✅ B2 · real-time tick loop + replay collection (`src/live_tick.py` · triggered by six decision-point kinds)
- ✅ B3 · low-latency decision LLM (`src/decision_llm.py` · 6 kinds via guided_json · ≤3s)
- ✅ B4 · WebSocket advice push service (`src/advice_server.py` · FastAPI)
- ✅ B5 · PyQt desktop overlay (`src/overlay_ui.py` · Win32 FindWindow + translucent card)

### C · Polish · portfolio-ready
- ✅ C1 · Hallucination audit upgrade · tokenization sliding window + whitelist
- ✅ C4 · Reports HTML renderer · replay reports written to disk as markdown/json/html
- ✅ C9 · README architecture diagram reworked into the dual-route form
- ✅ C10 · `scripts/e2e_smoke.py` end-to-end self-run in WSL · HTML replay in 50s
- ✅ C11 · `pitch/index.html` reworked into an AI-coach dual-entry landing page
- ✅ C12 · Positioning decode error fixed (pydantic field_validator tolerance)
- ✅ C13 · `scripts/setup_check.py` environment health check · 19 self-checks
- ✅ C14 · Windows one-click launch `scripts/start_coach.bat` (forwards to .ps1)

## Why pivot from boosting to analysis

The same tech stack (watch the screen · recognize UI · structure the state) · but different goals differ by an order of magnitude in value:

- Boosting → Tencent anti-cheat + drags down teammates' experience + no differentiation (real human boosters already exist) · gray area
- Analysis → players want to learn and climb · compliant · differentiates from existing "composition table" tools

Engineering-wise it's one thing · product-wise it's two.

## Author

[Xu Yunpeng](https://huannan.top) · systems designer + independent developer · Shanghai Jiao Tong Vocational and Technical College, class of 2026

Job target: big-tech AI / game design / agent applications

## License

MIT · see [LICENSE](./LICENSE)
