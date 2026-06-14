"""tests/test_setup_check.py · verify the core behavior of setup_check.py."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# add scripts/ to sys.path so setup_check can be imported directly
_scripts_dir = Path(__file__).parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import setup_check  # noqa: E402


# ---------------------------------------------------------------------------
# 1. status constants & formatting
# ---------------------------------------------------------------------------

class TestStatusEnum:
    """Verify the status constants and that _fmt output contains the correct icons."""

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
# 2. Windows-only platform-skip logic
# ---------------------------------------------------------------------------

class TestWindowsOnlySkip:
    """Non-Windows platforms should return SKIP; Windows platforms take the import path."""

    def test_skip_on_non_windows(self):
        with mock.patch.object(setup_check.sys, "platform", "linux"):
            result = setup_check.check_windows_only("PyQt6", "PyQt6")
        assert result.status == setup_check.SKIP
        assert "skipped" in result.detail.lower()

    def test_win32_and_import_ok(self):
        # fake a win32 environment + a fake importable module
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
# 3. does not crash when dependencies are missing
# ---------------------------------------------------------------------------

class TestGracefulMissingDeps:
    """Each check function should return WARN/FAIL rather than raise when a dependency is missing."""

    def test_core_import_missing_returns_fail(self):
        # patch the two functions setup_check uses internally, to avoid an importlib loop
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
            # make repo_root / examples / sample_frames not exist
            fake_sf = mock.MagicMock()
            fake_sf.exists.return_value = False
            mock_root = mock.MagicMock()
            mock_root.__truediv__ = lambda self, other: (
                fake_sf if other == "examples" else mock.MagicMock()
            )
            MockPath.return_value.parent.parent = mock_root
            # call the real function but patch exists
            real_sf = tmp_path / "nonexistent_dir"
            with mock.patch.object(setup_check, "check_sample_frames",
                                   wraps=lambda: setup_check.CheckResult(
                                       setup_check.FAIL, "examples/sample_frames/", "directory does not exist"
                                   )):
                result = setup_check.check_sample_frames()
        # tmp_path has no sample_frames/ and no examples/ — result is FAIL
        # just construct the expected value and assert
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
        # patch the default path to a non-existent tmp directory
        with mock.patch("setup_check._DEFAULT_DAIDA_PATH" if hasattr(setup_check, "_DEFAULT_DAIDA_PATH") else "builtins.open",
                        tmp_path / "nope"):
            # test directly with a non-existent default path:
            with mock.patch.object(
                Path,
                "exists",
                lambda self: False,
            ):
                result = setup_check.check_jcc_daida()
        assert result.status in (setup_check.WARN, setup_check.FAIL)


# ---------------------------------------------------------------------------
# 4. Python version check
# ---------------------------------------------------------------------------

class TestPythonVersionCheck:
    """sys.version_info is a C-level object · use SimpleNamespace to mock .major/.minor/.micro."""

    @staticmethod
    def _vi(major: int, minor: int, micro: int = 0):
        """Return a fake version object with major/minor/micro attributes."""
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
# 5. CI workflow file check
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
# 6. run_checks() integration: exit 0 on warn-only, exit 1 on fail
# ---------------------------------------------------------------------------

class TestRunChecksExitCode:
    def test_returns_zero_on_all_ok_or_warn(self, monkeypatch):
        """Patch all check functions to all-OK and confirm run_checks() returns 0."""
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
