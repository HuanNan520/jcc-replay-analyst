"""手动验证 OBS 虚拟摄像头接入。

前置：
  1. OBS Studio 已启动
  2. 在 OBS 里添加 "Window Capture" 或 "Game Capture" 抓 MuMu 窗口
  3. 按 OBS 右下角 "Start Virtual Camera" 启动虚拟摄像头

跑：
  python scripts/test_obs_capture.py --out /tmp/obs_test.png
"""
import argparse
import asyncio
import logging
from pathlib import Path

from src.capture_obs import OBSCapture


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/obs_test.png"))
    ap.add_argument("--count", type=int, default=3, help="抓几帧验证稳定性")
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with OBSCapture(fps=args.fps) as cap:
        for i in range(args.count):
            b = cap.read_once()
            path = args.out.with_name(f"{args.out.stem}_{i:02d}.png")
            path.write_bytes(b)
            print(f"✓ frame {i} · {len(b):,} bytes · saved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
