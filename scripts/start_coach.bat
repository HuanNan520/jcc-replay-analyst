@echo off
:: One-click launcher for the JCC real-time coach (cmd-compatible entry point)
:: Internally invokes start_coach.ps1 in the same directory, forwarding all arguments.
:: Usage:
::   start_coach.bat
::   start_coach.bat -SkipVLLM
::   start_coach.bat -ModelPath "/home/huannan/jcc-ai/models/Qwen3-VL-8B"
::   start_coach.bat -Help
powershell -ExecutionPolicy Bypass -File "%~dp0start_coach.ps1" %*
