"""Manually verify the OBS Virtual Camera connection.

Prerequisites:
  1. OBS Studio is running
  2. add a "Window Capture" or "Game Capture" source in OBS to capture the MuMu window
  3. click "Start Virtual Camera" at the bottom-right of OBS to start the virtual camera

Run:
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
    ap.add_argument("--count", type=int, default=3, help="number of frames to grab to verify stability")
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
