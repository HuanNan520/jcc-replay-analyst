"""jcc-replay-analyst · 金铲铲对局 AI 复盘分析。

只读屏幕 · 不操作游戏 · 合规。

组件：
  frame_monitor  从录屏/截图序列里抽取关键帧
  ocr_client     PaddleOCR · 读中文数字与 UI 文字
  arrow_finder   OpenCV · 找屏幕高亮 UI 元素（如教程引导框）
  adb_client     ADB · 仅做截屏（screencap / screenrecord fallback）
  vlm_client     Qwen VLM · 识别棋盘/阵容/羁绊等语义字段
  schema         WorldState / Unit / ActiveTrait 数据结构
  analyzer       把各层串起来 · 输出结构化对局报告
"""
