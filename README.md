# jcc-replay-analyst

[![CI](https://github.com/HuanNan520/jcc-replay-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/HuanNan520/jcc-replay-analyst/actions/workflows/ci.yml)

> 把金铲铲对局录屏丢给 AI · 让它告诉你每个关键节点做对还是做错。

<p align="center">
  <em>只读屏 · 不操作游戏 · 合规对局复盘工具</em>
</p>

## 要解决的问题

市面金铲铲工具只查**阵容表** —— 告诉你"这套强"。没人告诉你**你这局哪步错了**。

这个项目补空白：喂一局录屏 · AI 像教练一样指出你每个关键决策的得失。

## 示例报告

📄 [`examples/sample_report.md`](./examples/sample_report.md) — AI 生成的 markdown 报告样例

🎨 [`pitch/index.html`](./pitch/index.html) — 完整视觉呈现版（在浏览器打开）

## 技术架构

```
 一局录屏
    │
    ▼
 frame_monitor   抽出关键帧 · 每局 ~34 张 · 省 95% 冗余计算
    │
    ├─────────────┬─────────────┐
    ▼             ▼             ▼
 ocr_client   arrow_finder   vlm_client
 PaddleOCR    OpenCV         Qwen2.5-VL
 HP/金币/等级 棋子/装备/UI   羁绊/阵容/语义
    │             │             │
    └─────────────┼─────────────┘
                  ▼
            WorldState
         （结构化状态序列）
                  │
                  ▼
            analyzer · LLM
       （Claude API / 本地 · 金铲铲版本 RAG）
                  │
                  ▼
           MatchReport · Markdown
```

**混合架构的设计理由** · 让专精模型做专精事：

| 层 | 做什么 | 为什么不交给 VLM |
| --- | --- | --- |
| OCR · PaddleOCR | 读 HP / 金币 / 等级 中文数字 | Qwen2.5-VL 读 HP 常错 1-2 · OCR 稳定 99%+ |
| CV · OpenCV | 找装备图标 · 高亮 UI 元素 | 颜色/形状匹配 30 行代码够用 · 快且准 |
| VLM · Qwen2.5-VL | 羁绊语义 · 阵容分类 | VLM 该做的事 · 不让它读数字 |
| LLM · Claude / 本地 | 生成评分与叙事 | 需要金铲铲版本知识库 RAG |

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

### 运行时依赖 · 本地推理红线

**本项目 100% 本地推理 · 零云 API key**。

- 感知层（VLM）走本地 vLLM · OpenAI 兼容接口
- 分析层（LLM）同样走本地 vLLM · 默认和感知层复用同一实例和模型
- **严禁**引入 `anthropic` / `openai` / 任何云 SDK —— 仅用 `httpx` 直打 `/v1/chat/completions`
- 结构化输出优先使用 vLLM 原生 `guided_json` · 降级到 `response_format: json_object`
- 没有网 · 没有 key · 没有"云 fallback" —— 本地起不来就直接报错

## 项目状态

**Work in progress** · 当前状态：
- ✅ 感知层（frame_monitor · ocr · vlm）已跑通 · 对真实金铲铲画面识别稳定
- ✅ pipeline 骨架 · mock 模式下能跑完整流程出 markdown 报告
- ✅ LLM 分析层已接本地 vLLM · 产出带量化反事实的真实复盘（见 `src/llm_analyzer.py`）
- 🚧 金铲铲版本知识库（阵容/装备/羁绊 Meta）基础版已有 · 持续补充中

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
