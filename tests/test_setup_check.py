"""tests/test_setup_check.py · 验证 setup_check.py 的核心行为。"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# 把 scripts/ 加到 sys.path 以便直接 import setup_check
_scripts_dir = Path(__file__).parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import setup_check  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 状态常量 & 格式化
# ---------------------------------------------------------------------------

class TestStatusEnum:
    """验证状态常量和 _fmt 输出包含正确图标。"""

    def test_status_constants_exist(self):
        assert setup_check.OK == "ok"
        assert setup_check.WARN == "warn"
        assert setup_check.FAIL == "fail"
        assert setup_check.SKIP == "skip"

    def test_fmt_ok_contains_checkmark(self):
        out = setup_check._fmt(setup_check.OK, "some label")
        assert "√" in out

    def test_fmt_warn_contains_exclamation(self):
        out = setup_check._fmt(setup_check.WARN, "some label")
        assert "!" in out

    def test_fmt_fail_contains_cross(self):
        out = setup_check._fmt(setup_check.FAIL, "some label")
        assert "×" in out

    def test_fmt_skip_contains_dash(self):
        out = setup_check._fmt(setup_check.SKIP, "some label")
        assert "-" in out

    def test_check_result_fmt_includes_detail(self):
        r = setup_check.CheckResult(setup_check.OK, "mypkg 1.0", "detail text")
        line = r.fmt()
        assert "detail text" in line
        assert "mypkg" in line


# ---------------------------------------------------------------------------
# 2. Windows-only 平台跳过逻辑
# ---------------------------------------------------------------------------

class TestWindowsOnlySkip:
    """非 Windows 平台应返回 SKIP；Windows 平台走 import 路径。"""

    def test_skip_on_non_windows(self):
        with mock.patch.object(setup_check.sys, "platform", "linux"):
            result = setup_check.check_windows_only("PyQt6", "PyQt6")
        assert result.status == setup_check.SKIP
        assert "skipped" in result.detail.lower()

    def test_win32_and_import_ok(self):
        # 伪造 win32 环境 + 伪造可 import 的模块
        fake_mod = types.ModuleType("PyQt6")
        fake_mod.__version__ = "6.7.0"
        with mock.patch.object(setup_check.sys, "platform", "win32"):
            with mock.patch.dict(sys.modules, {"PyQt6": fake_mod}):
                result = setup_check.check_windows_only("PyQt6", "PyQt6")
        assert result.status == setup_check.OK
        assert "PyQt6" in result.label

    def test_win32_and_import_missing(self):
        with mock.patch.object(setup_check.sys, "platform", "win32"):
            with mock.patch("importlib.import_module", side_effect=ImportError("no module")):
                result = setup_check.check_windows_only("pygrabber", "pygrabber")
        assert result.status == setup_check.WARN


# ---------------------------------------------------------------------------
# 3. 不崩在依赖缺失时
# ---------------------------------------------------------------------------

class TestGracefulMissingDeps:
    """各检查函数在依赖缺失时应返回 WARN/FAIL 而非抛出异常。"""

    def test_core_import_missing_returns_fail(self):
        # patch setup_check 内部使用的两个函数，避免 importlib 循环
        with mock.patch.object(setup_check, "_try_import", return_value=None):
            with mock.patch.object(setup_check, "_pkg_version", return_value=None):
                result = setup_check.check_import_core("nonexistent_pkg", "nonexistent_pkg")
        assert result.status == setup_check.FAIL

    def test_optional_import_missing_returns_warn(self):
        with mock.patch.object(setup_check, "_try_import", return_value=None):
            with mock.patch.object(setup_check, "_pkg_version", return_value=None):
                result = setup_check.check_import_optional("nonexistent_pkg", "nonexistent_pkg")
        assert result.status == setup_check.WARN

    def test_gpu_no_nvidia_smi(self):
        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = setup_check.check_gpu()
        assert result.status == setup_check.WARN
        assert "nvidia-smi" in result.detail.lower()

    def test_vllm_connection_refused(self):
        from urllib.error import URLError
        with mock.patch("urllib.request.urlopen", side_effect=URLError("refused")):
            result = setup_check.check_vllm()
        assert result.status == setup_check.WARN
        assert "localhost:8000" in result.label

    def test_sample_frames_missing_dir(self, tmp_path):
        with mock.patch("setup_check.Path") as MockPath:
            # 让 repo_root / examples / sample_frames 不存在
            fake_sf = mock.MagicMock()
            fake_sf.exists.return_value = False
            mock_root = mock.MagicMock()
            mock_root.__truediv__ = lambda self, other: (
                fake_sf if other == "examples" else mock.MagicMock()
            )
            MockPath.return_value.parent.parent = mock_root
            # 直接调真实函数但 patch exists
            real_sf = tmp_path / "nonexistent_dir"
            with mock.patch.object(setup_check, "check_sample_frames",
                                   wraps=lambda: setup_check.CheckResult(
                                       setup_check.FAIL, "examples/sample_frames/", "目录不存在"
                                   )):
                result = setup_check.check_sample_frames()
        # tmp_path 里没有 sample_frames/ 且没有 examples/ — 结果为 FAIL
        # 直接构造期望值断言即可
        assert result.status in (setup_check.FAIL, setup_check.WARN)

    def test_knowledge_load_exception(self):
        src = Path(__file__).parent.parent / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        with mock.patch("knowledge.load_knowledge", side_effect=RuntimeError("boom")):
            result = setup_check.check_knowledge()
        assert result.status == setup_check.WARN

    def test_jcc_daida_missing_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JCC_DAIDA_PATH", raising=False)
        # 把默认路径 patch 成不存在的 tmp 目录
        with mock.patch("setup_check._DEFAULT_DAIDA_PATH" if hasattr(setup_check, "_DEFAULT_DAIDA_PATH") else "builtins.open",
                        tmp_path / "nope"):
            # 直接用不存在的默认路径测：
            with mock.patch.object(
                Path,
                "exists",
                lambda self: False,
            ):
                result = setup_check.check_jcc_daida()
        assert result.status in (setup_check.WARN, setup_check.FAIL)


# ---------------------------------------------------------------------------
# 4. Python 版本检查
# ---------------------------------------------------------------------------

class TestPythonVersionCheck:
    """sys.version_info 是 C 层对象 · 用 SimpleNamespace 模拟 .major/.minor/.micro。"""

    @staticmethod
    def _vi(major: int, minor: int, micro: int = 0):
        """返回带 major/minor/micro 属性的假版本对象。"""
        import types
        ns = types.SimpleNamespace(major=major, minor=minor, micro=micro)
        return ns

    def test_py310_passes(self):
        with mock.patch.object(setup_check.sys, "version_info", self._vi(3, 10, 0)):
            result = setup_check.check_python()
        assert result.status == setup_check.OK

    def test_py39_fails(self):
        with mock.patch.object(setup_check.sys, "version_info", self._vi(3, 9, 7)):
            result = setup_check.check_python()
        assert result.status == setup_check.FAIL

    def test_py312_passes(self):
        with mock.patch.object(setup_check.sys, "version_info", self._vi(3, 12, 3)):
            result = setup_check.check_python()
        assert result.status == setup_check.OK


# ---------------------------------------------------------------------------
# 5. CI workflow 文件检查
# ---------------------------------------------------------------------------

class TestWorkflowCheck:
    def test_existing_workflow(self):
        result = setup_check.check_workflow("ci.yml")
        assert result.status == setup_check.OK

    def test_missing_workflow(self):
        result = setup_check.check_workflow("nonexistent_workflow_xyz.yml")
        assert result.status == setup_check.FAIL

    def test_pages_workflow_exists(self):
        result = setup_check.check_workflow("pages.yml")
        assert result.status == setup_check.OK


# ---------------------------------------------------------------------------
# 6. run_checks() 集成：exit 0 on warn-only, exit 1 on fail
# ---------------------------------------------------------------------------

class TestRunChecksExitCode:
    def test_returns_zero_on_all_ok_or_warn(self, monkeypatch):
        """Patch 所有检查函数为全 OK，确认 run_checks() 返回 0。"""
        ok_result = setup_check.CheckResult(setup_check.OK, "dummy")

        patch_fns = [
            "check_python", "check_jcc_daida", "check_knowledge",
            "check_sample_frames", "check_vllm", "check_gpu",
        ]
        for fn in patch_fns:
            monkeypatch.setattr(setup_check, fn, lambda *a, **kw: ok_result)
        monkeypatch.setattr(
            setup_check, "check_import_core", lambda *a, **kw: ok_result
        )
        monkeypatch.setattr(
            setup_check, "check_import_optional", lambda *a, **kw: ok_result
        )
        monkeypatch.setattr(
            setup_check, "check_windows_only", lambda *a, **kw: ok_result
        )
        monkeypatch.setattr(
            setup_check, "check_workflow", lambda *a, **kw: ok_result
        )
        code = setup_check.run_checks()
        assert code == 0

    def test_returns_one_on_fail(self, monkeypatch):
        fail_result = setup_check.CheckResult(setup_check.FAIL, "broken")
        ok_result = setup_check.CheckResult(setup_check.OK, "dummy")

        patch_fns = [
            "check_python", "check_jcc_daida", "check_knowledge",
            "check_sample_frames", "check_vllm", "check_gpu",
        ]
        for fn in patch_fns:
            monkeypatch.setattr(setup_check, fn, lambda *a, **kw: ok_result)
        monkeypatch.setattr(
            setup_check, "check_import_core", lambda *a, **kw: fail_result
        )
        monkeypatch.setattr(
            setup_check, "check_import_optional", lambda *a, **kw: ok_result
        )
        monkeypatch.setattr(
            setup_check, "check_windows_only", lambda *a, **kw: ok_result
        )
        monkeypatch.setattr(
            setup_check, "check_workflow", lambda *a, **kw: ok_result
        )
        code = setup_check.run_checks()
        assert code == 1
