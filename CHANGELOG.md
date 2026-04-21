# CHANGELOG

按时间倒序 · 每行对应一个 commit · 只记 feat / 架构级 doc。零散 docstring 同步不列。

---

## 2026-04-22

### 阶段 C · 打磨与作品集化（8 项完成）

- `b25d07c` docs · README 示例段 HTML 置顶 · 真 LLM 产出优先于老 mock
- `5827e04` docs · `src/__init__.py` 13 模块分层定位刷新
- `b89c762` docs · `tasks/README.md` 全部任务完成状态同步
- `db9159a` docs · `pitch/roadmap.html` C 阶段从 planned 改 done
- `efc2686` chore · reports/ 目录标准化 · gitkeep + gitignore
- `91baddb` docs · README 项目状态段 · B 全绿 + C 摘要
- `c413341` feat · **C14** Windows 一键启动 `start_coach.bat` / `.ps1`
- `b8ec6bf` feat · **C12** positioning decode fix + **C13** 环境自检脚本
- `d557edf` ci · pages workflow 改 manual + README 顶部加 live URL
- `c03b583` ci · **C10.5** GitHub Pages workflow 部署 pitch + roadmap + sample
- `2a00d38` docs · **C11** `pitch/index.html` 翻新为 AI 教练双入口 landing（1451 行）
- `3c1511c` feat · **C9** README 双路径架构图 + **C10** e2e smoke 脚本
- `e360322` refactor · **C1** hallucinate audit 白名单扩充 · 滑动窗口收紧
- `7db0e0c` feat · **C1** hallucinate audit tokenization + **C4** report HTML 渲染器

### 阶段 B · 实时 coach pipeline 全链路

- `866ffe3` docs · 加 `pitch/roadmap.html` 工程路线图
- `755bded` feat · **B2** live_tick + **B5** PyQt overlay
- `ae97e2c` feat · **B1** OBS 虚拟摄像头 + **B3** 决策 LLM + **B4** WebSocket server · **S17 knowledge** 适配 (73 英雄 / 10 阵容 / top4_rate 排序)

### 阶段 A · MVP · 录屏复盘

- `36475b7` docs · 本地/云措辞软化 · 支持 fork 接云
- `60fd039` feat · **A1-A4** 接本地 vLLM Qwen guided_json · 12 帧 sample · S16 RAG · 40 pytest + CI
- `465d302` initial · pipeline 骨架 + pitch demo

---

## 数据（commit `b25d07c` 时点）

- **代码量**: 8226 LOC（src + tests + scripts）
- **测试**: 198 pytest passing · 零 regression
- **CI**: GitHub Actions 20+ 次 run 全绿
- **产物**: main/README landing + 3 个 HTML 展示页 + e2e smoke demo
- **知识库**: jcc-daida S17 · 73 英雄 · 37 羁绊 · 591 装备 · 10 阵容（按真 top4_rate）
- **决策延迟**: 本地 Qwen3-VL-4B-FP8 · 6 类 guided_json · 3-5s
