"""ADB glue layer — pull screenshots from an Android emulator / physical device.

Screencap only · never sends any touch/input command — this project only watches the
screen · it never operates the game.

- screencap        standard channel · works for the vast majority of frames
- screencap_via_record  fallback for games whose secure surface blocks screencap
                   (record 1s with screenrecord -> extract first frame with ffmpeg)
- screencap_retry  automatically falls back to the latter when the former fails · with reconnect fallback
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MUMU_DEFAULT_PORTS = [16384, 16416, 16448, 7555, 5555]

_ADB_CANDIDATES = [
    "adb",
    os.path.expanduser("~/.local/platform-tools/adb"),
    "/usr/bin/adb",
    "/usr/local/bin/adb",
]


def _locate_adb() -> str:
    for c in _ADB_CANDIDATES:
        if os.path.isabs(c):
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        else:
            found = shutil.which(c)
            if found:
                return found
    return "adb"


class ADBError(RuntimeError):
    pass


class ADBClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 16384, adb_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.device = f"{host}:{port}"
        self.adb = adb_path or _locate_adb()

    async def _run(self, *args: str, check: bool = True) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            self.adb, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if check and proc.returncode != 0:
            raise ADBError(f"adb {' '.join(args)} failed: {err.decode(errors='ignore')}")
        return out

    async def connect(self) -> bool:
        try:
            out = await self._run("connect", self.device, check=False)
            txt = out.decode(errors="ignore")
            ok = "connected" in txt.lower() and "cannot" not in txt.lower() and "failed" not in txt.lower()
            log.info("adb connect %s → %s", self.device, txt.strip())
            return ok
        except FileNotFoundError:
            raise ADBError("adb not installed. Run: sudo apt install -y android-tools-adb")

    async def auto_connect(self) -> bool:
        """Try the common emulator ports (MuMu / Nox / etc.) in turn."""
        for port in [self.port] + [p for p in MUMU_DEFAULT_PORTS if p != self.port]:
            self.port = port
            self.device = f"{self.host}:{port}"
            if await self.connect():
                log.info("auto-connect succeeded on port %d", port)
                return True
        return False

    async def screencap(self, save_path: Optional[Path] = None) -> bytes:
        data = await self._run("-s", self.device, "exec-out", "screencap", "-p")
        if not data or len(data) < 100:
            raise ADBError(
                f"screencap returned {len(data)} bytes — the emulator may be switching screens / blacked out / crashed"
            )
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ADBError(f"screencap did not return a PNG (first 8 bytes {data[:8]!r})")
        if save_path is not None:
            save_path.write_bytes(data)
        return data

    async def screencap_retry(
        self,
        save_path: Optional[Path] = None,
        max_retries: int = 2,
        reconnect_on_fail: bool = True,
    ) -> bytes:
        """Fault-tolerant screencap: try screencap first · fall back to screenrecord on an empty packet · auto-reconnect on network failure."""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self.screencap(save_path=save_path)
            except ADBError as e:
                last_err = e
                log.info("screencap failed on attempt %d (%s), trying the screenrecord fallback",
                         attempt + 1, str(e)[:80])
                try:
                    return await self.screencap_via_record(save_path=save_path)
                except Exception as re:
                    last_err = re
                    log.warning("screenrecord also failed: %s", re)
            except Exception as e:
                last_err = e
                if reconnect_on_fail:
                    try:
                        await self.auto_connect()
                    except Exception as ce:
                        log.warning("reconnect failed: %s", ce)
                await asyncio.sleep(0.5 * (attempt + 1))
        raise ADBError(f"screencap failed {max_retries} times in a row: {last_err}")

    async def screencap_via_record(
        self,
        save_path: Optional[Path] = None,
        duration_s: int = 1,
        bit_rate: str = "8M",
    ) -> bytes:
        """Record duration_s seconds with screenrecord -> extract the first frame with ffmpeg, return PNG.
        Bypasses the secure-surface screencap block of some games. Cost: about 1-2s slower than screencap.
        """
        import tempfile

        device_mp4 = "/sdcard/replay_cap.mp4"
        await self._run(
            "-s", self.device, "shell",
            "screenrecord", "--time-limit", str(duration_s),
            "--bit-rate", bit_rate,
            device_mp4,
        )
        with tempfile.TemporaryDirectory() as td:
            local_mp4 = os.path.join(td, "cap.mp4")
            local_png = os.path.join(td, "cap.png")
            await self._run("-s", self.device, "pull", device_mp4, local_mp4)
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", local_mp4,
                "-frames:v", "1", local_png,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()
            if rc != 0 or not os.path.isfile(local_png):
                raise ADBError("ffmpeg failed to extract a frame, make sure ffmpeg is available")
            with open(local_png, "rb") as f:
                data = f.read()

        if save_path is not None:
            save_path.write_bytes(data)
        return data

    async def get_resolution(self) -> tuple[int, int]:
        out = await self._run("-s", self.device, "shell", "wm", "size")
        txt = out.decode(errors="ignore")
        for part in txt.split():
            if "x" in part and part.replace("x", "").isdigit():
                w, h = part.split("x")
                return int(w), int(h)
        raise ADBError(f"cannot parse resolution: {txt}")

    async def devices(self) -> str:
        out = await self._run("devices", "-l")
        return out.decode(errors="ignore")
