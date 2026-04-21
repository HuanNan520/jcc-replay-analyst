"""HTML 报告渲染器 —— MatchReport → self-contained HTML 字符串。

视觉风格延续 pitch/index.html 和 pitch/roadmap.html：
- 深色背景 + 金色 accent + 朱砂 / 青瓷 / 赭石语义色
- 宋体衬线标题
- 半透明背景 + 噪点纹理（SVG data-uri）
- 响应式（手机切单列）
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
import html

if TYPE_CHECKING:
    pass

from .schema import MatchReport


# ──────────────────────────────────────────────────────────────────────────────
# CSS（完整内联 · 延续 pitch/index.html 的 CSS 变量语言）
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """\
:root {
  --bg: #0b0912;
  --bg-2: #13111c;
  --bg-3: #1b1826;
  --bg-4: #232032;
  --border: #2a2538;
  --border-soft: #1e1b2a;

  --ink: #e8e5d8;
  --ink-2: #a39d8e;
  --ink-3: #6b6458;
  --ink-4: #43403a;

  --gold: #c9a45d;
  --gold-bright: #e6c17a;
  --gold-soft: #8b7244;
  --gold-dim: rgba(201,164,93,0.14);
  --gold-glow: rgba(201,164,93,0.4);

  --vermilion: #b3432e;
  --vermilion-bright: #d55a3f;
  --vermilion-soft: rgba(179,67,46,0.18);

  --celadon: #5a8b7a;
  --celadon-soft: rgba(90,139,122,0.15);

  --ochre: #9c7a3c;
  --ochre-soft: rgba(156,122,60,0.15);

  --font-serif-cn: "Songti SC","STSong","Source Han Serif SC","Noto Serif CJK SC","SimSun","FangSong",serif;
  --font-serif-en: "Baskerville","Libre Baskerville","Hoefler Text",Georgia,"Times New Roman",serif;
  --font-sans: "PingFang SC","Hiragino Sans GB","Microsoft YaHei","Helvetica Neue",Arial,system-ui,sans-serif;
  --font-mono: "SF Mono","JetBrains Mono","Menlo","Consolas","Courier New",monospace;
}

*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}

body{
  background:var(--bg);
  color:var(--ink);
  font-family:var(--font-sans);
  font-size:15px;
  line-height:1.75;
  font-feature-settings:"palt" 1;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
  overflow-x:hidden;
}

body::before{
  content:'';
  position:fixed;inset:0;
  background:
    radial-gradient(ellipse 1200px 800px at 18% 12%,rgba(201,164,93,0.07),transparent 55%),
    radial-gradient(ellipse 900px 600px at 88% 80%,rgba(179,67,46,0.045),transparent 60%),
    radial-gradient(ellipse 800px 500px at 50% 50%,rgba(90,139,122,0.025),transparent 70%);
  pointer-events:none;z-index:0;
}
body::after{
  content:'';
  position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180' viewBox='0 0 180 180'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.85 0 0 0 0 0.78 0 0 0 0 0.6 0 0 0 0.14 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  opacity:0.55;mix-blend-mode:overlay;pointer-events:none;z-index:1;
}

.page-wrap{position:relative;z-index:2;max-width:880px;margin:0 auto;padding:40px 48px 80px;}

::selection{background:var(--gold);color:var(--bg)}

/* ── Header ── */
.page-header{
  display:flex;align-items:center;justify-content:space-between;
  padding-bottom:24px;margin-bottom:40px;
  border-bottom:1px solid var(--border-soft);
}
.brand{display:flex;align-items:center;gap:14px;font-family:var(--font-serif-cn);}
.brand-seal{
  display:inline-flex;align-items:center;justify-content:center;
  width:36px;height:36px;
  background:var(--vermilion);color:#f5e7d8;
  font-size:20px;font-weight:400;
  box-shadow:0 0 0 1px var(--vermilion),inset 0 0 0 2px rgba(255,255,255,0.12);
  transform:rotate(-2deg);
}
.brand-text{font-size:17px;color:var(--gold);letter-spacing:0.2em;}
.page-meta{font-family:var(--font-mono);font-size:11px;color:var(--ink-3);letter-spacing:0.1em;}

/* ── Hero / KPI ── */
.hero{
  background:linear-gradient(180deg,var(--bg-2),var(--bg-3));
  border:1px solid var(--border);
  padding:44px 48px 40px;
  position:relative;
  margin-bottom:40px;
}
.hero::before{
  content:'';position:absolute;top:-1px;left:-1px;
  width:64px;height:64px;
  border-top:2px solid var(--gold);border-left:2px solid var(--gold);
}
.hero::after{
  content:'';position:absolute;bottom:-1px;right:-1px;
  width:64px;height:64px;
  border-bottom:2px solid var(--gold);border-right:2px solid var(--gold);
}

.hero-stamp{
  position:absolute;top:28px;right:40px;
  width:76px;height:76px;
  background:var(--vermilion);color:#f5e7d8;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-family:var(--font-serif-cn);
  transform:rotate(-4deg);
  box-shadow:inset 0 0 0 3px rgba(255,230,200,0.15),inset 0 0 0 4px var(--vermilion),inset 0 0 0 5px rgba(255,230,200,0.15);
  opacity:0.9;
}
.hero-stamp-top{font-size:14px;letter-spacing:0.1em;margin-bottom:2px;}
.hero-stamp-bot{font-size:10px;letter-spacing:0.1em;opacity:0.85;}

.hero-title-row{
  display:flex;align-items:baseline;gap:16px;
  margin-bottom:28px;padding-bottom:20px;
  border-bottom:1px solid var(--border);
}
.hero-kind{font-family:var(--font-serif-cn);font-size:12px;color:var(--gold-soft);letter-spacing:0.3em;}
.hero-match-id{font-family:var(--font-mono);font-size:18px;color:var(--ink);letter-spacing:0.05em;}

.hero-body{
  display:grid;
  grid-template-columns:1fr auto;
  gap:40px;
  margin-bottom:0;
}
.hero-meta{display:flex;flex-direction:column;gap:12px;}
.meta-row{
  display:grid;grid-template-columns:90px 1fr;
  gap:12px;font-size:13px;align-items:baseline;
}
.meta-key{
  color:var(--ink-3);letter-spacing:0.18em;font-size:11px;text-align:right;
}
.meta-val{
  color:var(--ink);font-family:var(--font-mono);font-size:13px;
  padding-bottom:5px;border-bottom:1px dotted var(--border);
}

.rank-card{
  min-width:140px;text-align:center;
  padding:18px 28px;
  border:1px solid var(--gold-soft);
  background:linear-gradient(180deg,rgba(201,164,93,0.06),rgba(201,164,93,0.01));
}
.rank-label{font-size:10px;letter-spacing:0.35em;color:var(--gold-soft);margin-bottom:8px;text-transform:uppercase;}
.rank-big{
  font-family:var(--font-serif-en);font-size:64px;line-height:1;
  color:var(--gold-bright);letter-spacing:-0.04em;font-weight:400;
}
.rank-big span{font-size:20px;color:var(--gold-soft);margin-left:2px;letter-spacing:0;}
.rank-hp{margin-top:8px;font-size:11px;color:var(--ink-2);letter-spacing:0.15em;}

/* ── Key Rounds ── */
.section-head{
  font-family:var(--font-serif-cn);font-size:20px;color:var(--gold);
  margin-bottom:20px;padding-bottom:12px;
  border-bottom:1px solid var(--border);
  letter-spacing:0.08em;
}
.section-head::before{content:'§ ';}

.rounds-list{display:flex;flex-direction:column;}

.round-item{
  display:grid;grid-template-columns:80px 1fr auto;
  gap:24px;padding:22px 4px;
  border-bottom:1px dotted var(--border);
  align-items:flex-start;
  transition:background 0.25s;
}
.round-item:hover{background:rgba(201,164,93,0.025);}
.round-item:last-child{border-bottom:none;}

/* grade stamp */
.grade-stamp{
  width:72px;height:72px;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--font-serif-cn);font-size:36px;font-weight:400;
  position:relative;border:2px solid currentColor;
  transform:rotate(-3deg);flex-shrink:0;
}
.grade-stamp::before{
  content:'';position:absolute;inset:4px;
  border:1px solid currentColor;opacity:0.5;
}
.grade-stamp.grade-优{color:var(--celadon);background:var(--celadon-soft);}
.grade-stamp.grade-可{color:var(--ochre);background:var(--ochre-soft);}
.grade-stamp.grade-差{color:var(--vermilion);background:var(--vermilion-soft);}

.round-body{}
.round-head-row{display:flex;align-items:baseline;gap:14px;margin-bottom:8px;}
.round-when{font-family:var(--font-mono);font-size:18px;color:var(--gold);letter-spacing:0.05em;}
.round-what{font-family:var(--font-serif-cn);font-size:17px;color:var(--ink);letter-spacing:0.04em;}
.round-comment{font-size:14px;line-height:1.85;color:var(--ink-2);max-width:560px;}

.round-delta{
  font-family:var(--font-mono);font-size:11px;color:var(--ink-3);
  text-align:right;letter-spacing:0.1em;
  padding-top:4px;white-space:nowrap;
}
.round-delta b{
  display:block;font-family:var(--font-serif-cn);font-size:18px;
  color:var(--ink);font-weight:400;margin-top:4px;letter-spacing:0.05em;
}

/* empty rounds */
.rounds-empty{
  font-size:14px;color:var(--ink-3);
  padding:24px 0;font-style:italic;
}

/* ── Summary ── */
.summary-block{
  margin-top:40px;padding:32px 40px;
  background:linear-gradient(180deg,rgba(201,164,93,0.05),rgba(201,164,93,0.01));
  border:1px solid var(--gold-soft);
  border-left:3px solid var(--gold);
  position:relative;
}
.summary-label{
  font-size:10px;letter-spacing:0.4em;color:var(--gold);
  margin-bottom:14px;text-transform:uppercase;
}
.summary-text{
  font-family:var(--font-serif-cn);font-size:18px;line-height:1.95;
  color:var(--ink);letter-spacing:0.02em;
}

/* ── Footer ── */
.page-footer{
  margin-top:56px;padding-top:24px;
  display:flex;justify-content:space-between;align-items:center;
  font-size:11px;color:var(--ink-3);letter-spacing:0.15em;
  border-top:1px solid var(--border-soft);
}
.page-footer a{color:var(--gold-soft);text-decoration:none;}
.page-footer a:hover{color:var(--gold);}

/* ── Responsive ── */
@media(max-width:700px){
  .page-wrap{padding:24px 20px 60px;}
  .hero{padding:32px 24px 28px;}
  .hero-body{grid-template-columns:1fr;}
  .hero-stamp{top:14px;right:16px;width:58px;height:58px;}
  .hero-stamp-top{font-size:11px;}
  .hero-stamp-bot{font-size:9px;}
  .rank-big{font-size:48px;}
  .round-item{grid-template-columns:64px 1fr;}
  .round-delta{grid-column:2;text-align:left;}
  .grade-stamp{width:58px;height:58px;font-size:28px;}
  .summary-block{padding:24px 20px;}
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# CSS class 映射（grade → CSS class · 用 Unicode 转义让 CSS 匹配中文）
# ──────────────────────────────────────────────────────────────────────────────
_GRADE_CSS = {
    "优": "grade-优",
    "可": "grade-可",
    "差": "grade-差",
}


def _e(text: str) -> str:
    """HTML 实体转义。"""
    return html.escape(str(text), quote=False)


def render_report_html(report: MatchReport) -> str:
    """MatchReport → 完整 self-contained HTML 字符串（CSS 内联）。"""

    # 格式化时长
    m, s = divmod(report.duration_s, 60)
    h, m = divmod(m, 60)
    if h:
        duration_str = f"{h}h {m:02d}m {s:02d}s"
    else:
        duration_str = f"{m}m {s:02d}s"

    # 生成时间
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Hero 区 meta 行
    meta_rows_html = ""
    meta_rows_html += _meta_row("对局编号", report.match_id)
    if report.rank_tier:
        meta_rows_html += _meta_row("段位", report.rank_tier)
    meta_rows_html += _meta_row("核心阵容", report.core_comp or "未识别")
    meta_rows_html += _meta_row("对局时长", duration_str)
    meta_rows_html += _meta_row("最终血量", f"{report.final_hp} HP")

    # Key Rounds
    if report.key_rounds:
        rounds_html = ""
        for rr in report.key_rounds:
            grade_cls = _GRADE_CSS.get(rr.grade, "grade-可")
            delta_col = ""
            if rr.delta:
                delta_col = f'<div class="round-delta"><b>{_e(rr.delta)}</b></div>'
            rounds_html += f"""\
<div class="round-item">
  <div class="grade-stamp {grade_cls}">{_e(rr.grade)}</div>
  <div class="round-body">
    <div class="round-head-row">
      <span class="round-when">{_e(rr.round)}</span>
      <span class="round-what">{_e(rr.title)}</span>
    </div>
    <div class="round-comment">{_e(rr.comment)}</div>
  </div>
  {delta_col}
</div>
"""
    else:
        rounds_html = '<div class="rounds-empty">（无关键回合 · LLM 未识别出转折点）</div>'

    # Summary
    # 把换行变成段落
    summary_paras = [p.strip() for p in report.summary.split("\n") if p.strip()]
    if not summary_paras:
        summary_paras = [report.summary]
    summary_html = "".join(
        f'<p class="summary-text">{_e(p)}</p>' for p in summary_paras
    )

    # 报告编号印章文字（从 match_id 末尾取一小节）
    stamp_text = report.match_id[-4:] if len(report.match_id) >= 4 else report.match_id

    html_out = f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>对局复盘 · {_e(report.match_id)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="page-wrap">

  <header class="page-header">
    <div class="brand">
      <span class="brand-seal">鑑</span>
      <span class="brand-text">铲铲复盘</span>
    </div>
    <div class="page-meta">对局复盘报告 &nbsp;·&nbsp; {_e(gen_time)}</div>
  </header>

  <!-- ─── Hero ─── -->
  <section class="hero">
    <div class="hero-stamp">
      <span class="hero-stamp-top">复盘</span>
      <span class="hero-stamp-bot">{_e(stamp_text)}</span>
    </div>

    <div class="hero-title-row">
      <span class="hero-kind">对 局 复 盘</span>
      <span class="hero-match-id">{_e(report.match_id)}</span>
    </div>

    <div class="hero-body">
      <div class="hero-meta">
        {meta_rows_html}
      </div>
      <div class="rank-card">
        <div class="rank-label">最 终 排 名</div>
        <div class="rank-big">{report.final_rank}<span>th</span></div>
        <div class="rank-hp">HP {report.final_hp}</div>
      </div>
    </div>
  </section>

  <!-- ─── Key Rounds ─── -->
  <section style="margin-bottom:40px;">
    <h2 class="section-head">关键回合</h2>
    <div class="rounds-list">
      {rounds_html}
    </div>
  </section>

  <!-- ─── Summary ─── -->
  <section>
    <div class="summary-block">
      <div class="summary-label">AI 总评</div>
      {summary_html}
    </div>
  </section>

  <!-- ─── Footer ─── -->
  <footer class="page-footer">
    <span>生成时间：{_e(gen_time)}</span>
    <span><a href="https://github.com/HuanNan520/jcc-replay-analyst" target="_blank">by jcc-replay-analyst</a></span>
  </footer>

</div>
</body>
</html>"""

    return html_out


def _meta_row(key: str, val: str) -> str:
    return (
        f'<div class="meta-row">'
        f'<span class="meta-key">{_e(key)}</span>'
        f'<span class="meta-val">{_e(val)}</span>'
        f'</div>\n'
    )


__all__ = ["render_report_html"]
