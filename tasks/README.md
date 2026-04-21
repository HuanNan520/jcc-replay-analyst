# jcc-replay-analyst · 任务分配总览

---

## 已完成 · A1–A4（MVP · 录屏复盘）

| #  | 任务                         | 状态   | 关键产出 |
|----|------------------------------|--------|----------|
| A1 | 接本地 LLM 到 `_llm_synthesize` | ✅ 完成 | `src/llm_analyzer.py` · vLLM + Qwen guided_json |
| A2 | 补 examples/sample_frames/   | ✅ 完成 | 12 张真实 S16 帧 |
| A3 | 接 jcc-daida 作 S16 RAG      | ✅ 完成 | `src/knowledge.py` · 113 英雄 / 53 羁绊 / 10 阵容 |
| A4 | 起步测试套件 + CI            | ✅ 完成 | `tests/` × 40 passing · `.github/workflows/ci.yml` |

**MVP 状态**：clone 即可跑 mock pipeline · 本地 vLLM 起来可跑真 LLM · CI 绿。

---

## 产品定位升级 · 从复盘到实时教练

**新定位**：**实时 AI 教练**（主路线）+ **自动复盘**（副产品）

玩金铲铲的时候 · OBS 虚拟摄像头把游戏窗口推流 → Python 持续拉帧 → 感知层识别 → 决策点触发低延迟 LLM → 建议以半透明卡片形式悬浮在游戏窗口上方。对局结束自动合成整局复盘保存到 `reports/`。

### 三条红线

1. **只读屏不操作** —— 不发任何 ADB input / 键鼠模拟
2. **默认本地推理** —— 可 fork 接云 · 但默认零 API 成本
3. **不接 Android 侧** —— 走 OBS 虚拟摄像头 · 不走 ADB debugging · 腾讯反外挂感知不到这层

---

## B1–B5 · 实时教练 pipeline

| #  | 任务                         | 执行 AI             | 核心产出 | 依赖 |
|----|------------------------------|---------------------|----------|------|
| B1 | OBS 虚拟摄像头数据源         | **Sonnet 4.6**      | `src/capture_obs.py` · Windows 原生 Python | 无 |
| B2 | 实时 tick loop + 复盘归集    | **Opus 4.7**        | `src/live_tick.py` · 协调感知 → 决策 → 广播 | B1 · B3 · B4 |
| B3 | 低延迟决策 LLM（6 类）       | **Opus 4.7**        | `src/decision_llm.py` · 6 类短 prompt ≤3s | 无 |
| B4 | WebSocket 建议推送服务       | **Sonnet 4.6**      | `src/advice_server.py` · FastAPI + `/ws/advice` | 无 |
| B5 | PyQt 桌面 overlay            | **Opus 4.7**        | `src/overlay_ui.py` · Win32 FindWindow + 半透明卡片 | B4 |

### 并行策略

```
阶段 1 · 三窗口并开
  B1 (Sonnet) ── frames() stream            ─┐
  B3 (Opus)   ── DecisionLLM + 6 schemas    ─┤
  B4 (Sonnet) ── WebSocket broadcast        ─┘

阶段 2 · B2 (Opus) ── 组装 live_tick · 吃 B1/B3/B4

阶段 3 · B5 (Opus) ── PyQt overlay · 吃 B4 的 WS client
```

Opus 只能一个串行 · 所以三段次序：**B3 → B2 → B5**。Sonnet 的 B1/B4 在 B3 执行期间同时开着干。

---

## 通用约束（所有 B 任务）

1. **三条红线**见上 · 每个任务开头都自验
2. **YAGNI** —— 不做任务外的事 · 不提前抽象
3. **复用现有代码** —— 感知层 / schema / knowledge 基本零改 · 只加新文件
4. **Windows 原生 Python 代码** —— B1 / B5 明确要求 · B4 也最好能两边跑
5. **不 commit** —— 子代理只 diff · 不推 · 交给我审

---

## 仓库入口

已 clone 在 `~/jcc-replay-analyst`（WSL 侧）· A1-A4 产物已入 main 分支。

Windows 侧需要单独搞：

```powershell
# Windows PowerShell
cd C:\Users\huannan\Downloads\带走
git clone https://github.com/HuanNan520/jcc-replay-analyst.git jcc-replay-analyst-win
cd jcc-replay-analyst-win
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt   # B1/B5 会创建这个
```

---

## 任务文件

- [`B1_obs_capture.md`](./B1_obs_capture.md) — OBS 虚拟摄像头数据源
- [`B2_live_tick.md`](./B2_live_tick.md) — 实时 tick loop + 复盘归集
- [`B3_decision_llm.md`](./B3_decision_llm.md) — 低延迟决策 LLM
- [`B4_advice_server.md`](./B4_advice_server.md) — WebSocket 推送
- [`B5_overlay_ui.md`](./B5_overlay_ui.md) — PyQt 桌面 overlay

## 启动方式 · 和 A 阶段相同

```bash
# 三窗口并行
cd ~/jcc-replay-analyst
source .venv/bin/activate
claude
/model sonnet     # B1/B4
/model opus       # B3/B2/B5
```

贴第一条：
```
读 tasks/BX_XXX.md 的全部内容 · 严格按里面的约束和步骤执行 · 所有自验收清单逐条跑过全绿再回报我 · 不要动它范围外的文件 · 不要自己 git commit。
```
