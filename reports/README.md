# reports · match replay output directory

Once `src.live_tick` is running · each match end automatically composes a replay and drops three files here:

```
reports/TFT-S17-YYYYMMDD-HHMM.md     # markdown version · for an editor / blog
reports/TFT-S17-YYYYMMDD-HHMM.json   # raw MatchReport · for programmatic post-processing
reports/TFT-S17-YYYYMMDD-HHMM.html   # HTML version · dark visual, opens in a browser
```

For the HTML visual reference, see [`examples/sample_report.html`](../examples/sample_report.html).

This directory is governed by `.gitignore` — only `.gitkeep` and this README are committed · the actual report files are not.
