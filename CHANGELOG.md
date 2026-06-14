# CHANGELOG

Reverse chronological · one line per commit · only feat / architecture-level docs. Stray docstring syncs are not listed.

---

## 2026-04-22

### Phase C · Polish and portfolio-readiness (8 items done)

- `b25d07c` docs · README sample section moves HTML to the top · real LLM output takes priority over the old mock
- `5827e04` docs · `src/__init__.py` 13-module layered positioning refreshed
- `b89c762` docs · `tasks/README.md` synced to all-tasks-done status
- `db9159a` docs · `pitch/roadmap.html` phase C changed from planned to done
- `efc2686` chore · reports/ directory standardized · gitkeep + gitignore
- `91baddb` docs · README project-status section · B all green + C summary
- `c413341` feat · **C14** Windows one-click launch `start_coach.bat` / `.ps1`
- `b8ec6bf` feat · **C12** positioning decode fix + **C13** environment self-check script
- `d557edf` ci · pages workflow changed to manual + live URL added to top of README
- `c03b583` ci · **C10.5** GitHub Pages workflow deploys pitch + roadmap + sample
- `2a00d38` docs · **C11** `pitch/index.html` reworked into the AI-coach dual-entry landing (1451 lines)
- `3c1511c` feat · **C9** README dual-route architecture diagram + **C10** e2e smoke script
- `e360322` refactor · **C1** hallucination audit whitelist expanded · sliding window tightened
- `7db0e0c` feat · **C1** hallucination audit tokenization + **C4** report HTML renderer

### Phase B · Real-time coach pipeline, full chain

- `866ffe3` docs · added `pitch/roadmap.html` engineering roadmap
- `755bded` feat · **B2** live_tick + **B5** PyQt overlay
- `ae97e2c` feat · **B1** OBS Virtual Camera + **B3** decision LLM + **B4** WebSocket server · **S17 knowledge** adaptation (73 champions / 10 comps / top4_rate sorting)

### Phase A · MVP · screen-recording replay

- `36475b7` docs · softened the local/cloud wording · supports forking to a cloud backend
- `60fd039` feat · **A1-A4** hooked up local vLLM Qwen guided_json · 12-frame sample · S16 RAG · 40 pytest + CI
- `465d302` initial · pipeline skeleton + pitch demo

---

## Stats (as of commit `b25d07c`)

- **Code size**: 8226 LOC (src + tests + scripts)
- **Tests**: 198 pytest passing · zero regressions
- **CI**: GitHub Actions 20+ runs, all green
- **Artifacts**: main/README landing + 3 HTML showcase pages + e2e smoke demo
- **Knowledge base**: jcc-daida S17 · 73 champions · 37 traits · 591 items · 10 comps (by real top4_rate)
- **Decision latency**: local Qwen3-VL-4B-FP8 · 6 kinds via guided_json · 3-5s
