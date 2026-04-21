@echo off
:: JCC 实时 Coach 一键启动（cmd 兼容入口）
:: 内部调用同目录的 start_coach.ps1，支持透传所有参数。
:: 用法：
::   start_coach.bat
::   start_coach.bat -SkipVLLM
::   start_coach.bat -ModelPath "/home/huannan/jcc-ai/models/Qwen3-VL-8B"
::   start_coach.bat -Help
powershell -ExecutionPolicy Bypass -File "%~dp0start_coach.ps1" %*
