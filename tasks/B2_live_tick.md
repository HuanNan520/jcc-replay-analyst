# B2 · 实时 tick loop + 复盘归集

**分配给**：Claude Opus 4.7（`claude-opus-4-7`）· 核心架构 · 协调四个模块（B1 数据源 / 感知层 / B3 决策 / B4 广播）。
**依赖**：B1（帧流）· B3（决策 LLM）· B4（广播服务）**先完成**。
**预期工时**：1 天。
**运行时**：**Windows 原生 Python**（需要访问 OBS 虚拟摄像头）。
**新产品定位中的角色**：**实时 coach 的中央枢纽**。同时是复盘路线的新入口（对局结束自动调 `analyzer._llm_synthesize`）。

---

## 你是谁

你是被派到 `HuanNan520/jcc-replay-analyst` 执行 B2 的 Claude Opus 4.7。
B1/B3/B4 已经 merge（你开工前会确认）· 你的任务是把它们串成一个**闭环实时 coach**。

同时 · 你要把"对局结束 → 自动合成复盘"这条副线也接上 —— 复用现有 `src/llm_analyzer.py` · 不写新 LLM 代码。

## 数据流

```
OBSCapture.frames()                           ← B1
      ↓ bytes (PNG)
FrameMonitor.observe()                        ← 现有
      ↓ 关键帧触发
VLMClient.parse(bytes)                        ← 现有
      ↓ WorldState
_infer_decision_context(ws)                   ← 你写
      ↓ DecisionContext | None
                      ↓ None → 仅存 ring_buffer · 不叫 LLM
                      ↓ 有 → 调 B3
DecisionLLM.decide(ws, ctx)                   ← B3
      ↓ Advice
POST /advice to B4                            ← B4
      ↓ 也入 ring_buffer
                        
[对局结束 · stage == "end"]
      ↓
Analyzer._llm_synthesize(ring_buffer)         ← 现有 · 整局 WorldState 序列喂进来
      ↓
reports/S16-YYYYMMDD-HHMM.md + .json
```

## 目标产物

```bash
# 一条命令起实时 coach
python -m src.live_tick \
  --advice-server http://localhost:8765 \
  --llm-url http://localhost:8000/v1 \
  --llm-model Qwen3-VL-4B-FP8 \
  --fps 2 \
  --reports-dir reports/
```

---

## 具体要做

### 1. 新增 `src/live_tick.py`

```python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from .capture_obs import OBSCapture
from .frame_monitor import FrameMonitor, classify
from .vlm_client import VLMClient
from .ocr_client import find_number_near
from .schema import WorldState, MatchReport
from .decision_llm import DecisionLLM, DecisionContext, Advice
from .knowledge import load_knowledge
from .llm_analyzer import LocalLLMAnalyzer

log = logging.getLogger(__name__)


# ==================== Decision Context Inference ====================

def _infer_decision_context(ws: WorldState, prev_ws: Optional[WorldState]) -> Optional[DecisionContext]:
    """根据 WorldState 变化推断当前是否到了决策点 · 返回 DecisionContext 或 None。

    决策点判定规则（保守 · 宁可漏不可错判）：
    - augment: stage == 'augment' 且上一帧不是 augment（新弹出）· options 从 ws.augments 最后三项或空占位
    - carousel: stage == 'carousel' · options 从 ws.shop 5 个棋子名取
    - positioning: stage == 'positioning' · 单次触发
    - level: stage 切换到 'pve' / 'pvp' 刚开始 · 且 gold >= (level+1)*4（够升）
    - shop: 跳过 · 商店每回合都在 · 触发太频繁 · 等用户需求确认再加
    - item: bag 增到 >= 2 且出现新组件（变化检测）
    """
    if ws.stage == "augment" and (prev_ws is None or prev_ws.stage != "augment"):
        return DecisionContext(kind="augment", options=ws.augments[-3:] if ws.augments else ["?", "?", "?"])
    if ws.stage == "carousel" and (prev_ws is None or prev_ws.stage != "carousel"):
        return DecisionContext(kind="carousel", options=list(ws.shop[:5]))
    if ws.stage == "positioning" and (prev_ws is None or prev_ws.stage != "positioning"):
        return DecisionContext(kind="positioning", options=[])
    # level: 进入战斗回合开始时 · 且经济够升级门槛
    level_thresholds = {1: 0, 2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 96}
    if ws.stage in ("pve", "pvp") and (prev_ws is None or prev_ws.stage not in ("pve", "pvp")):
        required = level_thresholds.get(ws.level + 1, 999)
        if ws.gold >= required and ws.level < 9:
            return DecisionContext(kind="level", options=[])
    # item: bag 变多
    if prev_ws is not None and len(ws.bag) >= 2 and len(ws.bag) > len(prev_ws.bag):
        return DecisionContext(kind="item", options=[b.name for b in ws.bag])
    return None


# ==================== Advice Publisher ====================

class AdvicePublisher:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=3.0)
        return self

    async def __aexit__(self, *a):
        await self._client.aclose()

    async def publish(self, advice: Advice) -> None:
        try:
            r = await self._client.post(
                f"{self.server_url}/advice",
                json=advice.model_dump(),
            )
            r.raise_for_status()
            log.debug("advice published · %s · broadcast_to=%s", advice.kind, r.json().get("broadcast_to"))
        except Exception as e:
            log.warning("advice publish failed · %s · %s", advice.kind, e)


# ==================== Match Report Saver ====================

def _save_report(report: MatchReport, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md_path = reports_dir / f"{report.match_id}-{stamp}.md"
    json_path = md_path.with_suffix(".json")

    # 复用 scripts/analyze.py:report_to_markdown · 但那是 CLI 内部函数 · 简单复刻一份
    lines = [
        f"# 对局复盘 · {report.match_id}",
        "",
        f"- 最终排名：**{report.final_rank}** / 8",
        f"- 最终血量：{report.final_hp}",
        f"- 对局时长：{report.duration_s} 秒",
        f"- 核心阵容：{report.core_comp or '未识别'}",
        "",
        "## 关键回合",
        "",
    ]
    for r in report.key_rounds:
        lines += [
            f"### {r.round} · {r.title}",
            f"**评级：{r.grade}**" + (f"　{r.delta}" if r.delta else ""),
            "",
            r.comment,
            "",
        ]
    lines += ["## AI 总评", "", report.summary, ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info("复盘已保存 · %s", md_path)
    return md_path


# ==================== Live Tick Loop ====================

class LiveTickLoop:
    def __init__(
        self,
        capture: OBSCapture,
        vlm: VLMClient,
        decision_llm: DecisionLLM,
        post_match_llm: LocalLLMAnalyzer,
        publisher: AdvicePublisher,
        reports_dir: Path,
        ring_size: int = 60,
    ):
        self.capture = capture
        self.monitor = FrameMonitor(screen_size=(2560, 1456))
        self.vlm = vlm
        self.decision_llm = decision_llm
        self.post_match_llm = post_match_llm
        self.publisher = publisher
        self.reports_dir = reports_dir
        self.ring: deque[WorldState] = deque(maxlen=ring_size)
        self._prev_ws: Optional[WorldState] = None
        self._match_started = False

    async def run(self) -> None:
        log.info("LiveTickLoop 启动 · reports_dir=%s", self.reports_dir)
        async for frame in self.capture.frames():
            events = self.monitor.observe(frame)
            if not self.monitor.any_triggered(events) and self._match_started:
                continue   # 无关键帧变化 · 省 VLM 调用
            event_kind = classify(self.monitor.changed_regions(events))
            log.debug("frame event=%s", event_kind)

            try:
                ws = await self.vlm.parse(frame)
            except Exception as e:
                log.warning("VLM parse 失败 · %s", e)
                continue

            # 过滤无效状态（感知层可能给 unknown）
            if ws.stage == "unknown" and self._prev_ws is None:
                continue

            # 第一次见有效 stage · 标记对局开始
            if ws.stage != "unknown" and not self._match_started:
                self._match_started = True
                log.info("对局开始 · round=%s stage=%s", ws.round, ws.stage)

            self.ring.append(ws)

            # 对局结束 · 触发复盘
            if ws.stage == "end" and self._prev_ws is not None and self._prev_ws.stage != "end":
                await self._finalize_match()
                self._prev_ws = ws
                continue

            # 决策点判断
            ctx = _infer_decision_context(ws, self._prev_ws)
            if ctx:
                log.info("决策点触发 · kind=%s round=%s", ctx.kind, ws.round)
                try:
                    advice = await self.decision_llm.decide(ws, ctx)
                    await self.publisher.publish(advice)
                except Exception as e:
                    log.warning("决策链路失败 · %s", e)

            self._prev_ws = ws

    async def _finalize_match(self) -> None:
        log.info("对局结束 · 合成复盘 · ring size=%d", len(self.ring))
        try:
            report = await self.post_match_llm.synthesize(list(self.ring))
            _save_report(report, self.reports_dir)
        except Exception as e:
            log.error("复盘合成失败 · %s", e)
        self.ring.clear()
        self._match_started = False
        self._prev_ws = None


# ==================== CLI ====================

async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice-server", default="http://localhost:8765")
    ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
    ap.add_argument("--vlm-model", default="Qwen3-VL-4B-FP8")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="Qwen3-VL-4B-FP8")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    knowledge = load_knowledge()   # 默认 season='s17' · 实际赛季
    capture = OBSCapture(fps=args.fps)
    vlm = VLMClient(base_url=args.vlm_url, model=args.vlm_model, mode="real")
    decision_llm = DecisionLLM(base_url=args.llm_url, model=args.llm_model, knowledge=knowledge)
    post_match = LocalLLMAnalyzer(base_url=args.llm_url, model=args.llm_model, knowledge=knowledge)

    async with AdvicePublisher(args.advice_server) as publisher:
        loop = LiveTickLoop(
            capture=capture,
            vlm=vlm,
            decision_llm=decision_llm,
            post_match_llm=post_match,
            publisher=publisher,
            reports_dir=args.reports_dir,
        )
        await loop.run()


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
```

### 2. 单元测试 `tests/test_live_tick.py`

重点测 `_infer_decision_context` 的判定逻辑（无需真 LLM / 真 OBS）：

```python
import time
import pytest
from src.live_tick import _infer_decision_context
from src.schema import WorldState


def _ws(stage, round="1-1", hp=100, gold=0, level=1, bag=None):
    return WorldState(
        stage=stage, round=round, hp=hp, gold=gold, level=level,
        exp="0/0", timestamp=time.time(),
        bag=bag or [],
    )


class TestDecisionContextInference:
    def test_augment_first_entry(self):
        ctx = _infer_decision_context(_ws("augment"), prev_ws=None)
        assert ctx and ctx.kind == "augment"

    def test_augment_not_retriggered_if_still_augment(self):
        prev = _ws("augment")
        ctx = _infer_decision_context(_ws("augment"), prev_ws=prev)
        assert ctx is None

    def test_carousel_first_entry(self):
        ctx = _infer_decision_context(_ws("carousel"), prev_ws=None)
        assert ctx and ctx.kind == "carousel"

    def test_positioning_first_entry(self):
        ctx = _infer_decision_context(_ws("positioning"), prev_ws=_ws("pvp"))
        assert ctx and ctx.kind == "positioning"

    def test_level_triggered_when_gold_enough(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=10, level=3)  # 升 4 级要 10 金
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx and ctx.kind == "level"

    def test_level_not_triggered_when_insufficient_gold(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=4, level=3)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_level_capped_at_9(self):
        prev = _ws("augment")
        curr = _ws("pve", gold=500, level=9)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None

    def test_pve_to_pvp_not_retriggered(self):
        prev = _ws("pve", gold=10, level=3)
        curr = _ws("pvp", gold=8, level=4)
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None   # 已经在战斗回合里 · 不重复触发 level

    def test_no_trigger_on_pvp_steady(self):
        prev = _ws("pvp")
        curr = _ws("pvp")
        ctx = _infer_decision_context(curr, prev_ws=prev)
        assert ctx is None
```

### 3. 小调整：`Analyzer` 暴露的接口和 `LocalLLMAnalyzer` 复用

看 `src/analyzer.py` · 如果 `_llm_synthesize` 是 `Analyzer` 的方法 · 它实际调了 `LocalLLMAnalyzer.synthesize` —— 那你直接 `LocalLLMAnalyzer(...)` 实例化一样 · 上面代码里已是这个用法。

**不要改 `analyzer.py`** —— 复盘路径保持 CLI 入口可用。

---

## 禁止做的事

- 不要自己写 LLM 调用 · 直接用 B3 的 DecisionLLM + 现有 LocalLLMAnalyzer
- 不要改 `src/schema.py` · `src/vlm_client.py` · `src/llm_analyzer.py` · `src/knowledge.py` 任何一个
- 不要写"每帧打 VLM"的暴力版本 —— FrameMonitor 的关键帧触发是省算力核心
- 不要引入 queue.Queue / threading · 全程 asyncio
- 不要硬编码 screen_size · 用 FrameMonitor 的 auto orientation 推断（或者让 B1 返回分辨率）
- 不要写 daemon / supervisor 逻辑 —— 崩了就崩 · 上层用 shell systemd / nssm 守
- 不要持久化 ring buffer 到磁盘 · 完全内存 · 对局结束落报告
- 不要给每类 event 都写特殊分支 —— `classify` 的返回值目前只用来 debug log

---

## 自验收清单

- [ ] `python -c "from src.live_tick import LiveTickLoop, _infer_decision_context, AdvicePublisher, _save_report"` 导入无错
- [ ] `pytest tests/test_live_tick.py -v` 全绿 · 至少 9 个测试
- [ ] `python -m src.live_tick --help` 打印完整参数列表
- [ ] **集成测试 · 需要 B1/B3/B4 都跑着**：
  - 起 vLLM (8000) + advice_server (8765) + OBS 虚拟摄像头（开 MuMu）
  - 跑 `python -m src.live_tick --fps 1`
  - 进游戏选秀或选增强 · 看 `ws://localhost:8765/ws/advice` 有没有 advice 推来（用 websocat 验）
  - advice JSON 合法 · 含 kind/reasoning/confidence
- [ ] 对局走到 end stage · `reports/` 目录下出现 `.md` 和 `.json` 文件
- [ ] 跑完 40 个原 pytest + 9 个新 · 零回归
- [ ] `git diff --stat` 只含：
  - `src/live_tick.py` (新)
  - `tests/test_live_tick.py` (新)
  - `README.md` (可选 · 加一小段运行说明)

## 完成后

给用户 ≤ 200 字报告：
- 一次完整对局跑下来 · 触发了几类决策（augment × N / positioning × N / ...）
- 平均每帧处理耗时（FrameMonitor + VLM + 可能的 LLM）
- 对局结束 · 复盘文件路径 + 大小
- `_infer_decision_context` 的**假阴性 / 假阳性**观察（比如漏了某个关键节点 / 多触发一次）
- 给后续调优的 TODO 线索

不 git commit。

---

## 参考

- FrameMonitor.classify 的事件类型 · 看 `src/frame_monitor.py:216`
- A1 的 `src/llm_analyzer.py` 直接复用为 post-match analyzer · 不改
- B3 的 `DecisionLLM` / `DecisionContext` / `Advice` · 看 `src/decision_llm.py`
- B4 的 `/advice` endpoint body shape · 看 `src/advice_server.py`
