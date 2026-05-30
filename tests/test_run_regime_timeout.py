"""Test that run_regime() fails fast with a clear message on subprocess timeout."""

from __future__ import annotations

import pytest

from tests.conftest import run_regime


class TestRunRegimeTimeout:
    """Verify the subprocess timeout guard in run_regime()."""

    def test_timeout_raises_assertion_with_command_info(self):
        """run_regime() raises AssertionError naming the command on timeout.

        We exercise this by calling run_regime with args that cause the CLI
        to sleep indefinitely (--help exits immediately, so we use a bogus
        subcommand approach). Instead, we directly exercise the timeout by
        calling run_regime with a very short timeout and arguments that make
        the CLI take longer than that timeout.
        """
        # Use --version or --help won't work (exits fast). We need something
        # that actually runs the engine. Use a nonexistent CSV so the CLI
        # errors fast — BUT we need to trigger the timeout path directly.
        # The cleanest way: temporarily set timeout=1 and give a command
        # that actually starts a long-running process.
        #
        # We'll use a trick: pass a real CSV path but a very short timeout.
        # However, the engine might fail fast on small data too.
        #
        # Most robust: just call run_regime with a module that sleeps.
        # But run_regime hardcodes the module. So let's test the timeout
        # mechanism by temporarily monkeypatching subprocess.run to raise
        # TimeoutExpired.
        import subprocess
        from unittest.mock import patch

        fake_cmd = ["python", "-m", "hmm_futures_analysis.cli", "--csv", "test.csv"]
        with patch("tests.conftest.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=fake_cmd, timeout=1)
            with pytest.raises(AssertionError, match=r"timed out after 1s"):
                run_regime("--csv", "test.csv", timeout=1)

    def test_timeout_message_includes_engine_name(self):
        """The assertion message includes the full command for debugging."""
        import subprocess
        from unittest.mock import patch

        with patch("tests.conftest.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["anything"], timeout=1
            )
            with pytest.raises(
                AssertionError, match=r"hmm_futures_analysis.cli.*--engine.*fshmm"
            ):
                run_regime("--csv", "test.csv", "--engine", "fshmm", timeout=1)

    def test_no_timeout_returns_normally(self):
        """When the subprocess completes, run_regime returns the result."""
        import subprocess
        from unittest.mock import MagicMock, patch

        fake_result = MagicMock(spec=subprocess.CompletedProcess)
        fake_result.returncode = 0
        fake_result.stdout = "ok"
        fake_result.stderr = ""

        with patch("tests.conftest.subprocess.run", return_value=fake_result):
            result = run_regime("--csv", "test.csv", "--engine", "threshold")
            assert result.returncode == 0
            assert result.stdout == "ok"
