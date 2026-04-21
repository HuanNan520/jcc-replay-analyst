<#
.SYNOPSIS
    一键启动 JCC 实时 coach 全链路（vLLM + advice_server + live_tick + overlay_ui）

.DESCRIPTION
    分 4 步各开独立 PowerShell 窗口：
      1. WSL vLLM API server（验活后继续）
      2. Windows advice_server（WebSocket 广播）
      3. Windows live_tick（持续截帧 + LLM 决策）
      4. Windows overlay_ui（半透明建议卡片）

    核心逻辑说明（用于 review，不依赖外部测试）：
      - 验活用 curl 轮询，每 3 s 一次，最多 20 次（60 s）
      - 每个子窗口均设置窗口标题，便于识别
      - -SkipVLLM 跳过步骤 1，适合 vLLM 已在跑的情况
      - -ModelPath 覆盖默认模型路径

.PARAMETER SkipVLLM
    跳过 vLLM 启动（假定已在运行中）

.PARAMETER ModelPath
    覆盖默认模型路径（默认 /home/huannan/jcc-ai/models/Qwen3-VL-4B-FP8）

.PARAMETER Help
    显示用法并退出

.EXAMPLE
    .\start_coach.ps1
    .\start_coach.ps1 -SkipVLLM
    .\start_coach.ps1 -ModelPath "/home/huannan/jcc-ai/models/Qwen3-VL-8B"
    .\start_coach.bat -SkipVLLM
#>

param(
    [switch]$SkipVLLM,
    [string]$ModelPath = "/home/huannan/jcc-ai/models/Qwen3-VL-4B-FP8",
    [switch]$Help
)

# ──────────────────────────────────────────
# 用法
# ──────────────────────────────────────────
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# ──────────────────────────────────────────
# 颜色输出辅助
# ──────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "[STEP] $msg"  -ForegroundColor Cyan   }
function Write-OK    { param($msg) Write-Host "[ OK ] $msg"  -ForegroundColor Green  }
function Write-Warn  { param($msg) Write-Host "[WARN] $msg"  -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "[FAIL] $msg"  -ForegroundColor Red    }
function Write-Info  { param($msg) Write-Host "[INFO] $msg"  -ForegroundColor White  }

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "   JCC 实时 Coach · 一键启动                          " -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""

# ──────────────────────────────────────────
# 前置检查 1：当前目录是仓库根
# ──────────────────────────────────────────
Write-Step "检查工作目录..."
$repoCheck = Join-Path $PSScriptRoot "..\src\live_tick.py"
if (-not (Test-Path $repoCheck)) {
    Write-Fail "当前脚本不在仓库 scripts/ 目录，或 src/live_tick.py 不存在。"
    Write-Fail "请从仓库根或 scripts/ 目录运行此脚本。"
    exit 1
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-OK "仓库根：$repoRoot"

# ──────────────────────────────────────────
# 前置检查 2：Python 3 可用
# ──────────────────────────────────────────
Write-Step "检查 Python 3..."
try {
    $pyVer = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
    Write-OK "Python 版本：$pyVer"
} catch {
    Write-Fail "找不到 python 命令，请确认 Python 3 已安装并加入 PATH。"
    exit 1
}

# ──────────────────────────────────────────
# 前置检查 3：OBS Studio（warn-only）
# ──────────────────────────────────────────
Write-Step "检查 OBS Studio..."
$obsPath = "C:\Program Files\obs-studio\bin\64bit\obs64.exe"
$obsFound = $false
if (Test-Path $obsPath) {
    $obsFound = $true
    Write-OK "OBS Studio 找到：$obsPath"
} else {
    $obsCmd = Get-Command obs -ErrorAction SilentlyContinue
    if ($obsCmd) {
        $obsFound = $true
        Write-OK "OBS Studio 找到（PATH）：$($obsCmd.Source)"
    }
}
if (-not $obsFound) {
    Write-Warn "未检测到 OBS Studio（跳过，继续启动）。"
    Write-Warn "若需实时帧捕获，请先安装 OBS Studio。"
}

# ──────────────────────────────────────────
# 前置提示
# ──────────────────────────────────────────
Write-Host ""
Write-Host "┌─────────────────────────────────────────────────────┐" -ForegroundColor Yellow
Write-Host "│  务必先完成以下步骤再继续：                        │" -ForegroundColor Yellow
Write-Host "│  1. 打开 OBS Studio                                 │" -ForegroundColor Yellow
Write-Host "│  2. 添加 Window Capture 源 → 捕获 MuMu 模拟器窗口  │" -ForegroundColor Yellow
Write-Host "│  3. 右下角点击「Start Virtual Camera」              │" -ForegroundColor Yellow
Write-Host "└─────────────────────────────────────────────────────┘" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "已完成上述步骤？按 Enter 继续 · 输入 q 取消"
if ($confirm -eq 'q' -or $confirm -eq 'Q') {
    Write-Info "已取消。"
    exit 0
}

Write-Host ""

# ──────────────────────────────────────────
# 辅助：验活轮询（curl，最多 20 次，间隔 3 s = 60 s 超时）
# ──────────────────────────────────────────
function Wait-Service {
    param(
        [string]$Url,
        [string]$Label,
        [int]$MaxRetries = 20,
        [int]$IntervalSec = 3
    )
    Write-Info "等待 $Label 就绪（$Url）..."
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                Write-OK "$Label 已就绪（$i 次轮询）"
                return $true
            }
        } catch { }
        Write-Host "  [$i/$MaxRetries] 尚未就绪，3 s 后重试..." -ForegroundColor DarkGray
        Start-Sleep -Seconds $IntervalSec
    }
    Write-Warn "$Label 60 s 内未响应，继续启动后续步骤（可能导致链路不稳）。"
    return $false
}

# ──────────────────────────────────────────
# Step 1：WSL vLLM
# ──────────────────────────────────────────
if (-not $SkipVLLM) {
    Write-Step "Step 1/4 · 启动 WSL vLLM API Server..."

    $vllmCmd = "wsl -d Ubuntu -- bash -c `"~/jcc-ai/.venv/bin/python -m vllm.entrypoints.openai.api_server --model $ModelPath --served-model-name Qwen3-VL-4B-FP8 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192`""

    Start-Process powershell -ArgumentList `
        "-NoExit", `
        "-Command", `
        "`$host.UI.RawUI.WindowTitle = '[1] WSL vLLM · 8000'; $vllmCmd"

    Write-OK "vLLM 窗口已开启，等待模型加载..."
    Wait-Service -Url "http://localhost:8000/v1/models" -Label "vLLM" | Out-Null
} else {
    Write-Warn "Step 1 已跳过（-SkipVLLM），假定 vLLM 已在 localhost:8000 运行。"
}

Write-Host ""

# ──────────────────────────────────────────
# Step 2：advice_server
# ──────────────────────────────────────────
Write-Step "Step 2/4 · 启动 advice_server（Windows · 端口 8765）..."

$adviceCmd = "Set-Location '$repoRoot'; python -m src.advice_server --port 8765"

Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "`$host.UI.RawUI.WindowTitle = '[2] advice_server · 8765'; $adviceCmd"

Start-Sleep -Seconds 2
Wait-Service -Url "http://localhost:8765/health" -Label "advice_server" | Out-Null

Write-Host ""

# ──────────────────────────────────────────
# Step 3：live_tick
# ──────────────────────────────────────────
Write-Step "Step 3/4 · 启动 live_tick（Windows · fps=2）..."

$tickCmd = "Set-Location '$repoRoot'; python -m src.live_tick --fps 2 --advice-server http://localhost:8765 --llm-url http://localhost:8000/v1 --llm-model Qwen3-VL-4B-FP8"

Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "`$host.UI.RawUI.WindowTitle = '[3] live_tick · fps=2'; $tickCmd"

Write-OK "live_tick 窗口已开启。"

Write-Host ""

# ──────────────────────────────────────────
# Step 4：overlay_ui
# ──────────────────────────────────────────
Write-Step "Step 4/4 · 启动 overlay_ui（Windows · MuMu 贴边悬浮卡片）..."

$overlayCmd = "Set-Location '$repoRoot'; python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice"

Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "`$host.UI.RawUI.WindowTitle = '[4] overlay_ui · ws:8765'; $overlayCmd"

Write-OK "overlay_ui 窗口已开启。"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "   全链路启动完成！共 4 个终端窗口                    " -ForegroundColor Green
Write-Host "   [1] WSL vLLM        → localhost:8000               " -ForegroundColor White
Write-Host "   [2] advice_server   → localhost:8765               " -ForegroundColor White
Write-Host "   [3] live_tick       → 持续截帧 + 决策              " -ForegroundColor White
Write-Host "   [4] overlay_ui      → MuMu 窗口贴边悬浮            " -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "关闭各窗口即可停止对应服务。" -ForegroundColor DarkGray
