"""环境健康检查 · 跑一遍看实时 coach 所有前置是否就绪。

用法: python3 scripts/setup_check.py
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional
from urllib import request as urlrequest
from urllib.error import URLError

# ---------------------------------------------------------------------------
# 状态枚举 & 结果结构
# ---------------------------------------------------------------------------

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_STATUS_ICON = {OK: "√", WARN: "!", FAIL: "×", SKIP: "-"}
_STATUS_COLOR = {
    OK: "\033[32m",    # green
    WARN: "\033[33m",  # orange/yellow
    FAIL: "\033[31m",  # red
    SKIP: "\033[90m",  # grey
}
_RESET = "\033[0m"

_USE_COLOR = sys.stdout.isatty()


def _fmt(status: str, message: str) -> str:
    icon = _STATUS_ICON.get(status, "?")
    if _USE_COLOR:
        color = _STATUS_COLOR.get(status, "")
        return f"  {color}{icon}{_RESET} {message}"
    return f"  {icon} {message}"


@dataclass
class CheckResult:
    status: str          # OK / WARN / FAIL / SKIP
    label: str
    detail: str = ""

    def fmt(self) -> str:
        msg = self.label
        if self.detail:
            msg += f" · {self.detail}"
        return _fmt(self.status, msg)


@dataclass
class CheckGroup:
    title: str
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)

    def print(self) -> None:
        print(f"\n{self.title}")
        for r in self.results:
            print(r.fmt())


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _try_import(module: str) -> Optional[str]:
    """尝试 import 模块，返回版本字符串 or None。"""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if v and isinstance(v, str):
            return v
    try:
        return importlib.metadata.version(module)
    except Exception:
        pass
    return "installed"


def _pkg_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 各检查项
# ---------------------------------------------------------------------------

def check_python() -> CheckResult:
    vi = sys.version_info
    ver_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if (vi.major, vi.minor) >= (3, 10):
        return CheckResult(OK, f"Python {ver_str}")
    return CheckResult(FAIL, f"Python {ver_str}", "需要 >= 3.10")


def check_import_core(module: str, dist_name: str = "", display: str = "") -> CheckResult:
    label = display or dist_name or module
    v = _try_import(module)
    if v is None:
        # fallback: check via dist_name
        if dist_name:
            v = _pkg_version(dist_name)
    if v is None:
        return CheckResult(FAIL, label, "未安装 (pip install required)")
    short = ".".join(v.split(".")[:2]) if "." in v else v
    return CheckResult(OK, f"{label} {short}")


def check_import_optional(module: str, dist_name: str = "", display: str = "") -> CheckResult:
    label = display or dist_name or module
    v = _try_import(module)
    if v is None and dist_name:
        v = _pkg_version(dist_name)
    if v is None:
        return CheckResult(WARN, label, "未安装 (optional · 感知功能降级)")
    short = ".".join(v.split(".")[:2]) if "." in v else v
    return CheckResult(OK, f"{label} {short}")


def check_windows_only(module: str, display: str = "") -> CheckResult:
    label = display or module
    if sys.platform != "win32":
        return CheckResult(SKIP, label, f"Windows only · skipped on {sys.platform}")
    v = _try_import(module)
    if v is None:
        return CheckResult(WARN, label, "未安装 (Windows 端需要)")
    short = ".".join(v.split(".")[:2]) if "." in v else v
    return CheckResult(OK, f"{label} {short}")


def check_jcc_daida() -> CheckResult:
    env = os.environ.get("JCC_DAIDA_PATH")
    default = Path("/mnt/c/Users/huannan/Downloads/带走/jcc-daida")
    if env:
        p = Path(env).expanduser()
        if (p / "client.py").exists():
            return CheckResult(OK, f"jcc-daida at {p}")
        return CheckResult(FAIL, "jcc-daida", f"JCC_DAIDA_PATH={env} 但 client.py 不存在")
    if (default / "client.py").exists():
        return CheckResult(OK, f"jcc-daida at {default}")
    return CheckResult(WARN, "jcc-daida", "路径未找到 · 知识库功能不可用 (设 JCC_DAIDA_PATH)")


def check_knowledge() -> CheckResult:
    # 把项目 src 加入 sys.path 以便 import
    src = Path(__file__).parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from knowledge import load_knowledge  # type: ignore
        kb = load_knowledge()
    except Exception as e:
        return CheckResult(WARN, "load_knowledge()", f"调用失败 · {e}")
    if kb is None:
        return CheckResult(WARN, "load_knowledge()", "返回 None · jcc-daida 不可用")
    n_heroes = len(kb.all_units)
    n_comps = len(kb.comps)
    return CheckResult(OK, "load_knowledge()", f"{n_heroes} heroes · {n_comps} comps")


def check_sample_frames() -> CheckResult:
    repo_root = Path(__file__).parent.parent
    sf = repo_root / "examples" / "sample_frames"
    if not sf.exists():
        return CheckResult(FAIL, "examples/sample_frames/", "目录不存在")
    imgs = [
        f for f in sf.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ]
    n = len(imgs)
    if n < 5:
        return CheckResult(WARN, "examples/sample_frames/", f"只有 {n} 张图 (需要 >= 5)")
    return CheckResult(OK, "examples/sample_frames/", f"{n} frames")


def check_vllm() -> CheckResult:
    url = "http://localhost:8000/v1/models"
    try:
        req = urlrequest.Request(url, headers={"User-Agent": "setup-check/1.0"})
        with urlrequest.urlopen(req, timeout=2) as resp:
            body = resp.read()
        data = json.loads(body)
        models = [m.get("id", "?") for m in data.get("data", [])]
        if models:
            names = " / ".join(models[:3])
            return CheckResult(OK, f"vLLM at localhost:8000", f"模型: {names}")
        return CheckResult(OK, "vLLM at localhost:8000", "running (no models listed)")
    except (URLError, OSError):
        return CheckResult(WARN, "vLLM at localhost:8000", "connection refused (not running)")
    except Exception as e:
        return CheckResult(WARN, "vLLM at localhost:8000", f"error · {e}")


def check_gpu() -> CheckResult:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"],
            timeout=5,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return CheckResult(WARN, "GPU", "nvidia-smi 无输出")
        # 取第一行
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpu_name = parts[0]
            mem_free_str = parts[1]  # e.g. "13200 MiB"
            # 转 GB
            try:
                mib = float(mem_free_str.split()[0])
                gb = mib / 1024
                mem_str = f"{gb:.1f} GB free"
            except Exception:
                mem_str = mem_free_str
            return CheckResult(OK, f"GPU · {gpu_name}", mem_str)
        return CheckResult(OK, "GPU", out)
    except FileNotFoundError:
        return CheckResult(WARN, "GPU", "nvidia-smi 未找到 (无 NVIDIA GPU 或未安装驱动)")
    except subprocess.TimeoutExpired:
        return CheckResult(WARN, "GPU", "nvidia-smi 超时")
    except subprocess.CalledProcessError as e:
        return CheckResult(WARN, "GPU", f"nvidia-smi 失败 · {e}")


def check_workflow(name: str) -> CheckResult:
    repo_root = Path(__file__).parent.parent
    p = repo_root / ".github" / "workflows" / name
    if p.exists():
        return CheckResult(OK, f".github/workflows/{name}")
    return CheckResult(FAIL, f".github/workflows/{name}", "文件不存在")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_checks() -> int:
    today = date.today().isoformat()
    print(f"[setup-check · {today}] jcc-replay-analyst 环境自检")

    groups: list[CheckGroup] = []

    # --- Python · core ---
    g1 = CheckGroup("Python · core")
    g1.add(check_python())
    g1.add(check_import_core("pydantic", "pydantic", "pydantic"))
    g1.add(check_import_core("httpx", "httpx", "httpx"))
    g1.add(check_import_core("PIL", "Pillow", "Pillow"))
    g1.add(check_import_core("numpy", "numpy", "numpy"))
    groups.append(g1)

    # --- Perception · optional ---
    g2 = CheckGroup("Perception · optional")
    g2.add(check_import_optional("paddleocr", "paddleocr", "paddleocr"))
    g2.add(check_import_optional("cv2", "opencv-python", "opencv-python"))
    g2.add(check_windows_only("PyQt6", "PyQt6"))
    g2.add(check_windows_only("pygrabber", "pygrabber"))
    groups.append(g2)

    # --- Real-time coach · services ---
    g3 = CheckGroup("Real-time coach · services")
    g3.add(check_import_optional("fastapi", "fastapi", "fastapi"))
    g3.add(check_import_optional("uvicorn", "uvicorn", "uvicorn"))
    g3.add(check_import_optional("websockets", "websockets", "websockets"))
    groups.append(g3)

    # --- Knowledge · S17 ---
    g4 = CheckGroup("Knowledge · S17")
    g4.add(check_jcc_daida())
    g4.add(check_knowledge())
    groups.append(g4)

    # --- Sample data ---
    g5 = CheckGroup("Sample data")
    g5.add(check_sample_frames())
    groups.append(g5)

    # --- Runtime · optional ---
    g6 = CheckGroup("Runtime · optional")
    g6.add(check_vllm())
    g6.add(check_gpu())
    groups.append(g6)

    # --- CI ---
    g7 = CheckGroup("CI")
    g7.add(check_workflow("ci.yml"))
    g7.add(check_workflow("pages.yml"))
    groups.append(g7)

    # 打印所有组
    for g in groups:
        g.print()

    # 汇总
    all_results: list[CheckResult] = []
    for g in groups:
        all_results.extend(g.results)

    total = len(all_results)
    n_ok = sum(1 for r in all_results if r.status == OK)
    n_warn = sum(1 for r in all_results if r.status == WARN)
    n_fail = sum(1 for r in all_results if r.status == FAIL)
    n_skip = sum(1 for r in all_results if r.status == SKIP)

    warn_labels = [r.label for r in all_results if r.status == WARN]
    fail_labels = [r.label for r in all_results if r.status == FAIL]

    print("\nSummary")
    parts = [f"{total} checks", f"{n_ok} ok"]
    if n_skip:
        parts.append(f"{n_skip} skipped")
    if n_warn:
        warn_hint = " / ".join(warn_labels[:3])
        parts.append(f"{n_warn} warn ({warn_hint})")
    if n_fail:
        fail_hint = " / ".join(fail_labels[:3])
        parts.append(f"{n_fail} FAIL ({fail_hint})")

    print(f"  {' · '.join(parts)}")

    # exit 1 只有 FAIL 才触发
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(run_checks())
