"""jcc-replay-analyst · 金铲铲之战 AI 教练。

实时对局建议 + 自动赛后复盘 双入口。只读屏 · 不操作游戏 · 合规。

组件分层：

感知层（两路径共享）:
  frame_monitor  dHash 关键帧检测 · 按 ROI 分区
  ocr_client     PaddleOCR · 读中文数字 / UI 文字
  arrow_finder   OpenCV · 找屏幕高亮 UI 元素
  vlm_client     Qwen VLM · 识别棋盘/阵容/羁绊等语义 → WorldState
  schema         WorldState / Unit / ActiveTrait / MatchReport

数据源:
  adb_client     ADB 截屏（旧路线 · 仅复盘用）
  capture_obs    OBS 虚拟摄像头（实时路线 · 主推）

知识 RAG:
  knowledge      包 jcc-daida · S17 默认 · S16 向后兼容 · 为 LLM 注版本上下文

决策 + 推理:
  decision_llm   六类决策点的短 prompt · 本地 vLLM guided_json · ≤3s
  llm_analyzer   整局 WorldState 序列 → MatchReport · 本地 vLLM guided_json
  analyzer       复盘路径 pipeline 编排（吃录屏/截图序列 → MatchReport）
  live_tick      实时路径 tick loop · 协调感知 → 决策 → 广播 · 对局结束归集复盘

交付:
  advice_server  FastAPI + WebSocket · 建议广播
  overlay_ui     PyQt6 半透明卡片 · Win32 FindWindow 跟随 MuMu（Windows only）
  report_renderer  MatchReport → self-contained HTML（深色金色视觉）
"""
