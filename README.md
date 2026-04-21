# jcc-replay-analyst

[![CI](https://github.com/HuanNan520/jcc-replay-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/HuanNan520/jcc-replay-analyst/actions/workflows/ci.yml)

🌐 **在线访问** · [pitch landing](https://huannan520.github.io/jcc-replay-analyst/) · [工程路线图](https://huannan520.github.io/jcc-replay-analyst/roadmap.html) · [真 LLM 样例复盘](https://huannan520.github.io/jcc-replay-analyst/sample_report.html)（首次需 repo Settings → Pages → Source 选 "GitHub Actions" · 然后 Actions 标签手动跑 Deploy Pages）

> 金铲铲之战 AI 教练 · **实时对局建议** + **自动赛后复盘** 双入口。

<p align="center">
  <em>只读屏 · 不操作游戏 · OBS 虚拟摄像头驱动 · 合规</em>
</p>

## 要解决的问题

市面金铲铲工具只查**阵容表** —— 告诉你"这套强"。没人在**你玩的时候**告诉你**该选哪个增强 · 要不要升人口 · 装备合给谁** · 也没人事后告诉你**这局哪步错了**。

这个项目两头补：
- **实时 coach**：玩家玩的时候 · OBS 把 MuMu 窗口推成虚拟摄像头 · Python 持续识别 · 决策点触发 LLM · 建议以半透明卡片浮现在 MuMu 窗口上方
- **自动复盘**：对局结束自动合成整局复盘报告到 `reports/`（无需手动录屏）

同一套感知层（VLM + OCR + CV + S16 知识 RAG）驱动两条路线。

## 示例报告

📄 [`examples/sample_report.md`](./examples/sample_report.md) — AI 生成的 markdown 报告样例

📊 [`examples/sample_report.html`](./examples/sample_report.html) — AI 生成的 HTML 复盘报告（在浏览器打开 · 完整深色视觉风格）

🎨 [`pitch/index.html`](./pitch/index.html) — 完整视觉呈现版（在浏览器打开）

🗺️ [`pitch/roadmap.html`](./pitch/roadmap.html) — 工程路线图（A/B/C 阶段进度可视化）

## 端到端 smoke demo

不依赖真 OBS · 不依赖 Windows · 在 WSL 里跑通全链路 · 50 秒内出一份 HTML 复盘：

```bash
python3 scripts/e2e_smoke.py
# vLLM 可用时真跑 5 类 decisionLLM + 整局合成复盘
# vLLM 不可用时自动 fallback · 仍能跑通 pipeline
# 产出: /tmp/e2e_smoke_report.html (≈ 18 KB)
```

## 技术架构

```
                    数据源层（二选一）
   ┌──────────────────────┬──────────────────────┐
   │  OBS Virtual Camera  │  mp4 录屏 / 截图序列  │
   │  (实时 coach 路线)   │  (赛后复盘路线)      │
   └──────────┬───────────┴──────────┬───────────┘
              │                      │
              ▼                      ▼
      ┌───────────────────────────────────────┐
      │  共享感知层                           │
      │  frame_monitor (dHash 关键帧)          │
      │  ocr_client (PaddleOCR 数字)           │
      │  vlm_client (Qwen3-VL 语义)            │
      │  → WorldState                         │
      └──────────┬────────────────┬───────────┘
                 │                │
                 ▼                ▼
     ┌─────────────────┐  ┌──────────────────┐
     │  实时 coach     │  │  赛后复盘         │
     │  live_tick →    │  │  analyzer →       │
     │  decision_llm   │  │  llm_analyzer →   │
     │  (6 类 ≤3s) →   │  │  MatchReport →    │
     │  advice_server  │  │  md/json/html     │
     │  → overlay_ui   │  │  (复用实时路径的  │
     │  (半透明卡片)   │  │   对局结束归集)    │
     └─────────────────┘  └──────────────────┘

            S17 版本知识库（jcc-daida）·
            → KnowledgeProvider 注入两路径
```

**混合架构的设计理由** · 让专精模型做专精事：

| 层 | 做什么 | 为什么不交给 VLM |
| --- | --- | --- |
| OCR · PaddleOCR | 读 HP / 金币 / 等级 中文数字 | Qwen2.5-VL 读 HP 常错 1-2 · OCR 稳定 99%+ |
| CV · OpenCV | 找装备图标 · 高亮 UI 元素 | 颜色/形状匹配 30 行代码够用 · 快且准 |
| VLM · Qwen2.5-VL | 羁绊语义 · 阵容分类 | VLM 该做的事 · 不让它读数字 |
| LLM · 本地 vLLM | 生成评分与叙事 | 需要金铲铲版本知识库 RAG · 默认本地零 API 成本 |

**各层独立失败不互相污染** · 比押注单一大 VLM 模型稳得多。

## 模块说明

```
src/
├── adb_client.py      ADB 截屏（screencap / screenrecord fallback）· 不发任何输入指令
├── frame_monitor.py   dHash 感知哈希 · 按 ROI 检测屏幕变化 · 抽关键帧
├── ocr_client.py      PaddleOCR 封装 · recognize / find_number_near / 长气泡检测
├── arrow_finder.py    OpenCV · 找屏幕亮色 UI 元素中心
├── vlm_client.py      Qwen VLM · 识别画面语义字段 → WorldState
├── schema.py          WorldState / RoundReview / MatchReport 数据结构
└── analyzer.py        pipeline · 吃帧序列吐 MatchReport
scripts/
└── analyze.py         CLI 入口
```

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 纯骨架跑（mock VLM · mock LLM · 验证 pipeline）
python scripts/analyze.py --frames examples/sample_frames/ --vlm mock --llm mock --out report.md

# 3. 接真 VLM（需要本地 vLLM 跑 Qwen2.5-VL 或 Qwen3-VL）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --port 8000 &
python scripts/analyze.py --frames path/to/frames/ --vlm real --out report.md

# 4. 从 mp4 录屏直接抽帧分析
python scripts/analyze.py --video match.mp4 --every-s 5 --vlm real --out report.md

# 5. 接本地 LLM 分析层（分析和感知复用同一个 vLLM 实例）
#    显存 ≥ 14GB free 用 Qwen3-VL-8B-FP8 · 紧张（如 16GB 卡还跑了模拟器）降到 Qwen3-VL-4B-FP8
python scripts/analyze.py \
  --frames examples/sample_frames/ \
  --vlm mock --llm real \
  --llm-url http://localhost:8000/v1 \
  --llm-model Qwen3-VL-4B-FP8 \
  --out report.md
# 也可以 VLM + LLM 一起真跑 · 把 --vlm mock 换成 --vlm real 即可
```

### 运行时 · 默认本地推理

**默认走本地 vLLM · 零 API 成本 · 开箱即用**。不锁死本地 —— 想付费接云 API（Claude / OpenAI / 其他），写一个满足同样接口的 `CloudLLMAnalyzer` 替换即可。

- 感知层（VLM）走本地 vLLM · OpenAI 兼容接口
- 分析层（LLM）同样走本地 vLLM · 默认和感知层复用同一实例和模型
- 默认不引入 `anthropic` / `openai` 作为硬依赖（云 API 成本远高于自建跑 Qwen）· 仅用 `httpx` 直打 `/v1/chat/completions`
- 结构化输出优先使用 vLLM 原生 `guided_json` · 降级到 `response_format: json_object`
- 零网络依赖 · 开箱即本地跑 —— 迁云时只需另写一个 Analyzer 实现

### 实时 coach 模式 · 数据源

实时模式要求 Windows 原生 Python（不在 WSL） · 前置：

1. 装 OBS Studio 并启动
2. 添加 "Window Capture" 抓 MuMu 窗口
3. OBS 右下 "Start Virtual Camera"
4. `pip install -r requirements-windows.txt`
5. `python scripts/test_obs_capture.py` 验证接入

### 实时 coach 模式 · 完整启动（4 个终端 · B2 完成后）

```powershell
# 终端 1 · WSL · 启 vLLM（感知 + 分析 + 决策共用）
source ~/jcc-replay-analyst/.venv/bin/activate
python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-VL-4B-FP8 --port 8000

# 终端 2 · Windows 或 WSL · 启 advice 广播服务
python -m src.advice_server --port 8765

# 终端 3 · Windows 原生 Python · 起 OBS virtual cam + live tick（前置 OBS Start Virtual Camera）
python -m src.live_tick --fps 2 --advice-server http://localhost:8765

# 终端 4 · Windows 原生 Python · 起 overlay（半透明卡片浮在 MuMu 窗口上方）
python -m src.overlay_ui --ws-url ws://localhost:8765/ws/advice
```

overlay 会自动找 MuMu 主窗口并贴边 · 决策点触发就淡入卡片 · 8 秒后淡出。
开发调试时用 `--no-click-through` 可以点到 overlay 本身。

**或者一条命令**（Windows）：`scripts\start_coach.bat`（或 `.ps1`）· 自动开 4 个终端跑完整链路。

## 项目状态

**198 pytest passing · CI 绿 · 15 个任务 commit** · 三条路线：

### A · 复盘路线（MVP done）
- ✅ 感知层（frame_monitor · ocr · vlm）已跑通 · 对真实金铲铲画面识别稳定
- ✅ pipeline 骨架 · mock 模式能跑完整流程出 markdown 报告
- ✅ LLM 分析层已接本地 vLLM · 产出带量化反事实的真实复盘（`src/llm_analyzer.py`）
- ✅ S17 版本知识库接入 `jcc-daida` · 73 英雄 / 37 羁绊 / 591 装备 / 10 阵容按 top4_rate 排（`src/knowledge.py`）
- ✅ 40 个 pytest 绿 · GitHub Actions CI 跑通

### B · 实时 coach 路线（done · 见 `tasks/README.md`）
- ✅ B1 · OBS 虚拟摄像头数据源（`src/capture_obs.py` · 替代 ADB · 合规更稳）
- ✅ B2 · 实时 tick loop + 复盘归集（`src/live_tick.py` · 六类决策点触发）
- ✅ B3 · 低延迟决策 LLM（`src/decision_llm.py` · 6 类 guided_json · ≤3s）
- ✅ B4 · WebSocket 建议推送服务（`src/advice_server.py` · FastAPI）
- ✅ B5 · PyQt 桌面 overlay（`src/overlay_ui.py` · Win32 FindWindow + 半透明卡片）

### C · 打磨 · 作品集化
- ✅ C1 · Hallucinate audit 升级 · tokenization 滑动窗口 + 白名单
- ✅ C4 · Reports HTML 渲染器 · 复盘报告 markdown/json/html 三联落盘
- ✅ C9 · README 架构图翻新成双路径
- ✅ C10 · `scripts/e2e_smoke.py` 端到端 WSL 自跑 · 50s 出 HTML 复盘
- ✅ C11 · `pitch/index.html` 翻新为 AI 教练双入口 landing
- ✅ C12 · Positioning decode error 修（pydantic field_validator 容错）
- ✅ C13 · `scripts/setup_check.py` 环境健康检查 · 19 项自检
- ✅ C14 · Windows 一键启动 `scripts/start_coach.bat`（转发 .ps1）

## 为什么从代打转做分析

同一套技术栈（看屏幕 · 识别 UI · 结构化状态） · 目标不同价值差一个数量级：

- 代打 → 腾讯反外挂 + 拉低队友体验 + 无差异化（有真人代练）· 灰色地带
- 分析 → 玩家想学上分 · 合规 · 与现有"阵容表"类工具形成差异化

工程上是同一件事 · 产品上是两件事。

## 作者

[徐雲鵬](https://huannan.top) · 系统策划 + 独立开发者 · 上海交通职业技术学院 2026 届

求职方向：大厂 AI / 游戏策划 / Agent 应用

## License

MIT · 见 [LICENSE](./LICENSE)
