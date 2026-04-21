# reports · 对局复盘落盘目录

`src.live_tick` 跑起来后 · 每局结束会自动合成复盘并在此落三份文件：

```
reports/TFT-S17-YYYYMMDD-HHMM.md     # markdown 版 · 给编辑器/博客用
reports/TFT-S17-YYYYMMDD-HHMM.json   # 原始 MatchReport · 程序化二次处理
reports/TFT-S17-YYYYMMDD-HHMM.html   # HTML 版 · 浏览器打开即深色视觉
```

HTML 版视觉参考 [`examples/sample_report.html`](../examples/sample_report.html)。

本目录由 `.gitignore` 管控 —— 只保留 `.gitkeep` 和本说明进 git · 实际报告文件不入库。
