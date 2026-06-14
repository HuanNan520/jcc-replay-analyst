<#
.SYNOPSIS
    One-click launch of the full JCC real-time coach pipeline (vLLM + advice_server + live_tick + overlay_ui)

.DESCRIPTION
    Opens 4 separate PowerShell windows in 4 steps:
      1. WSL vLLM API server (continues after the health check)
      2. Windows advice_server (WebSocket broadcast)
      3. Windows live_tick (continuous frame capture + LLM decisions)
      4. Windows overlay_ui (semi-transparent advice cards)

    Core logic notes (for review, independent of external tests):
      - the health check polls with curl every 3 s, up to 20 times (60 s)
      - each child window sets a window title for easy identification
      - -SkipVLLM skips step 1, suitable when vLLM is already running
      - -ModelPath overrides the default model path

.PARAMETER SkipVLLM
    Skip the vLLM launch (assume it is already running)

.PARAMETER ModelPath
    Override the default model path (default /home/huannan/jcc-ai/models/Qwen3-VL-4B-FP8)

.PARAMETER Help
    Show usage and exit

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
# usage
# ──────────────────────────────────────────
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# ──────────────────────────────────────────
# colored-output helpers
# ──────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "[STEP] $msg"  -ForegroundColor Cyan   }
function Write-OK    { param($msg) Write-Host "[ OK ] $msg"  -ForegroundColor Green  }
function Write-Warn  { param($msg) Write-Host "[WARN] $msg"  -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "[FAIL] $msg"  -ForegroundColor Red    }
function Write-Info  { param($msg) Write-Host "[INFO] $msg"  -ForegroundColor White  }

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "   JCC Real-time Coach · One-click launch             " -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""

# ──────────────────────────────────────────
# Prerequisite check 1: current directory is the repo root
# ──────────────────────────────────────────
Write-Step "Checking working directory..."
$repoCheck = Join-Path $PSScriptRoot "..\src\live_tick.py"
if (-not (Test-Path $repoCheck)) {
    Write-Fail "This script is not in the repo scripts/ directory, or src/live_tick.py does not exist."
    Write-Fail "Please run this script from the repo root or the scripts/ directory."
    exit 1
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-OK "Repo root: $repoRoot"

# ──────────────────────────────────────────
# Prerequisite check 2: Python 3 is available
# ──────────────────────────────────────────
Write-Step "Checking Python 3..."
try {
    $pyVer = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
    Write-OK "Python version: $pyVer"
} catch {
    Write-Fail "The python command was not found; please make sure Python 3 is installed and on PATH."
    exit 1
}

# ──────────────────────────────────────────
# Prerequisite check 3: OBS Studio (warn-only)
# ──────────────────────────────────────────
Write-Step "Checking OBS Studio..."
$obsPath = "C:\Program Files\obs-studio\bin\64bit\obs64.exe"
$obsFound = $false
if (Test-Path $obsPath) {
    $obsFound = $true
    Write-OK "OBS Studio found: $obsPath"
} else {
    $obsCmd = Get-Command obs -ErrorAction SilentlyContinue
    if ($obsCmd) {
        $obsFound = $true
        Write-OK "OBS Studio found (PATH): $($obsCmd.Source)"
    }
}
if (-not $obsFound) {
    Write-Warn "OBS Studio not detected (skipping, continuing launch)."
    Write-Warn "If you need real-time frame capture, please install OBS Studio first."
}

# ──────────────────────────────────────────
# Prerequisite prompt
# ──────────────────────────────────────────
Write-Host ""
Write-Host "┌─────────────────────────────────────────────────────┐" -ForegroundColor Yellow
Write-Host "│  Be sure to complete the following before continuing:│" -ForegroundColor Yellow
Write-Host "│  1. Open OBS Studio                                 │" -ForegroundColor Yellow
Write-Host "│  2. Add a Window Capture source -> capture the MuMu  │" -ForegroundColor Yellow
Write-Host "│     emulator window                                 │" -ForegroundColor Yellow
Write-Host "│  3. Click 'Start Virtual Camera' at bottom-right    │" -ForegroundColor Yellow
Write-Host "└─────────────────────────────────────────────────────┘" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Completed the steps above? Press Enter to continue · type q to cancel"
if ($confirm -eq 'q' -or $confirm -eq 'Q') {
    Write-Info "Cancelled."
    exit 0
}

Write-Host ""

# ──────────────────────────────────────────
# Helper: health-check polling (curl, up to 20 times, 3 s interval = 60 s timeout)
# ──────────────────────────────────────────
function Wait-Service {
    param(
        [string]$Url,
        [string]$Label,
        [int]$MaxRetries = 20,
        [int]$IntervalSec = 3
    )
    Write-Info "Waiting for $Label to be ready ($Url)..."
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                Write-OK "$Label is ready ($i polls)"
                return $true
            }
        } catch { }
        Write-Host "  [$i/$MaxRetries] not ready yet, retrying in 3 s..." -ForegroundColor DarkGray
        Start-Sleep -Seconds $IntervalSec
    }
    Write-Warn "$Label did not respond within 60 s; continuing with the next steps (the pipeline may be unstable)."
    return $false
}

# ──────────────────────────────────────────
# Step 1: WSL vLLM
# ──────────────────────────────────────────
if (-not $SkipVLLM) {
    Write-Step "Step 1/4 · Starting the WSL vLLM API Server..."

    $vllmCmd = "wsl -d Ubuntu -- bash -c `"~/jcc-ai/.venv/bin/python -m vllm.entrypoints.openai.api_server --model $ModelPath --served-model-name Qwen3-VL-4B-FP8 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192`""

    Start-Process powershell -ArgumentList `
        "-NoExit", `
        "-Command", `
        "`$host.UI.RawUI.WindowTitle = '[1] WSL vLLM · 8000'; $vllmCmd"

    Write-OK "vLLM window opened, waiting for the model to load..."
    Wait-Service -Url "http://localhost:8000/v1/models" -Label "vLLM" | Out-Null
} else {
    Write-Warn "Step 1 skipped (-SkipVLLM), assuming vLLM is already running on localhost:8000."
}

Write-Host ""

# ──────────────────────────────────────────
# Step 2: advice_server
# ──────────────────────────────────────────
Write-Step "Step 2/4 · Starting advice_server (Windows · port 8765)..."

$adviceCmd = "Set-Location '$repoRoot'; python -m src.advice_server --port 8765"

Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "`$host.UI.RawUI.WindowTitle = '[2] advice_server · 8765'; $adviceCmd"

Start-Sleep -Seconds 2
Wait-Service -Url "http://localhost:8765/health" -Label "advice_server" | Out-Null

Write-Host ""

# ──────────────────────────────────────────
# Step 3: live_tick
# ──────────────────────────────────────────
Write-Step "Step 3/4 · Starting live_tick (Windows · fps=2)..."

$tickCmd = "Set-Location '$repoRoot'; python -m src.live_tick --fps 2 --advice-server http://localhost:8765 --llm-url http://localhost:8000/v1 --llm-model Qwen3-VL-4B-FP8"

Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "`$host.UI.RawUI.WindowTitle = '[3] live_tick · fps=2'; $tickCmd"

Write-OK "live_tick window opened."

Write-Host ""

# ──────────────────────────────────────────
# Step 4: overlay_ui
# ──────────────────────────────────────────
Write-Step "Step 4/4 · Starting overlay_ui (Windows · floating cards docked to MuMu)..."

$overlayCmd = "Set-Location '$repoRoot'; python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice"

Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "`$host.UI.RawUI.WindowTitle = '[4] overlay_ui · ws:8765'; $overlayCmd"

Write-OK "overlay_ui window opened."

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "   Full pipeline launched! 4 terminal windows total  " -ForegroundColor Green
Write-Host "   [1] WSL vLLM        -> localhost:8000              " -ForegroundColor White
Write-Host "   [2] advice_server   -> localhost:8765              " -ForegroundColor White
Write-Host "   [3] live_tick       -> continuous capture + decisions" -ForegroundColor White
Write-Host "   [4] overlay_ui      -> floating, docked to MuMu     " -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "Close each window to stop its corresponding service." -ForegroundColor DarkGray
