#!/usr/bin/env python3
"""CLI · generate a replay-analysis report for a screen-recording / screenshot sequence of one match.

Usage:
  # analyze from a screenshot directory (filenames in frame order)
  python scripts/analyze.py --frames data/sample_match/ --out report.md

  # analyze from an mp4 recording (automatic frame extraction)
  python scripts/analyze.py --video data/sample.mp4 --out report.md

  # use mock VLM/LLM only · no model service required · validate the pipeline
  python scripts/analyze.py --frames data/sample_match/ --vlm mock --llm mock
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# let `python scripts/analyze.py` import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import Analyzer, AnalyzerConfig


def iter_frame_bytes_from_dir(directory: Path):
    files = sorted(f for f in directory.iterdir()
                   if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        raise SystemExit(f"no image frames found under {directory}")
    for f in files:
        yield f.read_bytes()


def iter_frame_bytes_from_video(video: Path, every_s: float = 5.0):
    """Extract frames with ffmpeg · one every every_s seconds · yield bytes."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        out_pattern = Path(td) / "f%04d.png"
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vf", f"fps=1/{every_s}", str(out_pattern),
        ]
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc != 0:
            raise SystemExit("ffmpeg frame extraction failed · please make sure ffmpeg is available")
        for f in sorted(Path(td).glob("f*.png")):
            yield f.read_bytes()


def report_to_markdown(report) -> str:
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
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, help="screenshot directory · one frame per file")
    ap.add_argument("--video", type=Path, help="screen-recording mp4 · automatic frame extraction")
    ap.add_argument("--every-s", type=float, default=5.0, help="frame-extraction interval in seconds for video mode")
    ap.add_argument("--out", type=Path, default=Path("report.md"), help="report output path")
    ap.add_argument("--vlm", choices=["real", "mock"], default="mock")
    ap.add_argument("--llm", choices=["real", "mock"], default="mock")
    ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1",
                    help="local vLLM OpenAI-compatible URL · defaults to reusing the same instance as the VLM")
    ap.add_argument("--llm-model", default="Qwen3-VL-8B-FP8",
                    help="analysis-layer LLM model name · defaults to reusing the perception layer's Qwen3-VL-8B-FP8")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.frames and not args.video:
        ap.error("a --frames directory or a --video file is required")

    if args.frames:
        frames = iter_frame_bytes_from_dir(args.frames)
    else:
        frames = iter_frame_bytes_from_video(args.video, args.every_s)

    analyzer = Analyzer(AnalyzerConfig(
        vlm_base_url=args.vlm_url,
        vlm_mode=args.vlm,
        llm_mode=args.llm,
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
    ))

    report = await analyzer.analyze_frames(frames)
    md = report_to_markdown(report)
    args.out.write_text(md, encoding="utf-8")
    print(f"✓ report written: {args.out} ({len(md)} chars)")

    # also dump the raw JSON · convenient for further processing
    args.out.with_suffix(".json").write_text(
        report.model_dump_json(indent=2, by_alias=False),
        encoding="utf-8",
    )
    print(f"✓ json dump: {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    asyncio.run(main())
