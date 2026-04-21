# sample_frames

一局完整 S16 对局的关键帧摘录（12 张 · 覆盖选秀→选增强→战斗→结算全流程）。

供 `scripts/analyze.py --frames examples/sample_frames/` 作为 e2e smoke test。

帧按文件名顺序送入 `frame_monitor.observe` · 首帧建 baseline · 后续帧触发事件抽取。

| 帧 | 阶段 | 游戏回合 |
| --- | --- | --- |
| frame_001_pick | 选秀/开局 carousel | 1-1 |
| frame_002_pve | 小兵 PVE | 1-4 海滩 |
| frame_003_positioning | 备战/摆位 | 2-x |
| frame_004_pve | 小兵 PVE | 2-5 |
| frame_005_augment | 选增强 | 3-2 |
| frame_006_pvp | 玩家对战 | 3-4 |
| frame_007_pvp | 玩家对战 | 4-1 |
| frame_008_item | 装备转盘 | 4-4 |
| frame_009_positioning | 备战/摆位 | 5-1 |
| frame_010_pvp | 玩家对战 | 5-4 |
| frame_011_pvp | 玩家对战 | 6-1 |
| frame_012_end | 结算 | 第二名 |

源帧来自 `MuMu-20260421-08xxxx` 系列（2560×1456 原图）· 缩至 1280×720、JPEG q=85 压缩以控制体积（总计 ~2.2 MB）。
