"""jcc-replay-analyst · an AI coach for Teamfight Tactics (China-server mobile, jkchess).

Two entry points: real-time in-match advice + automatic post-match replay analysis.
Screen-read only · never controls the game · compliant.

Component layers:

Perception layer (shared by both paths):
  frame_monitor  dHash keyframe detection · partitioned by ROI
  ocr_client     PaddleOCR · reads Chinese numbers / UI text
  arrow_finder   OpenCV · locates highlighted UI elements on screen
  vlm_client     Qwen VLM · recognizes board / composition / traits semantically -> WorldState
  schema         WorldState / Unit / ActiveTrait / MatchReport

Data sources:
  adb_client     ADB screencap (legacy path · replay analysis only)
  capture_obs    OBS virtual camera (real-time path · primary)

Knowledge RAG:
  knowledge      wraps jcc-daida · S17 default · S16 backward compatible · injects version context for the LLM

Decision + reasoning:
  decision_llm   short prompts for six decision-point types · local vLLM guided_json · <=3s
  llm_analyzer   full-match WorldState sequence -> MatchReport · local vLLM guided_json
  analyzer       replay-path pipeline orchestration (recording/screenshot sequence -> MatchReport)
  live_tick      real-time tick loop · coordinates perception -> decision -> broadcast · assembles a replay when the match ends

Delivery:
  advice_server  FastAPI + WebSocket · advice broadcast
  overlay_ui     PyQt6 translucent card · Win32 FindWindow follows MuMu (Windows only)
  report_renderer  MatchReport -> self-contained HTML (dark gold visual theme)
"""
