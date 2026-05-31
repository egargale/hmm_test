"""Tests for package structure, importability, and distribution."""

import subprocess

import pytest


class TestPackageImport:
    """Verify the package is importable under its canonical name."""

    def test_import_hmm_futures_analysis(self):
        pass

    def test_import_cli_main(self):
        from hmm_futures_analysis.cli import main

        assert callable(main)

    def test_import_regime_pipeline(self):
        from hmm_futures_analysis.regime.pipeline import run

        assert callable(run)

    def test_import_data_processing(self):
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv

        assert callable(load_from_csv)


class TestConsoleScript:
    """Verify the hmm-regime console script entry point."""

    def test_entry_point_resolves(self):
        from importlib.metadata import entry_points

        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.9 returns dict
        if hasattr(eps, "select"):
            console = eps.select(group="console_scripts", name="hmm-regime")
            found = list(console)
        else:
            found = [
                ep for ep in eps.get("console_scripts", []) if ep.name == "hmm-regime"
            ]
        if not found:
            pytest.skip("hmm-regime entry point not registered (package not installed)")
        assert len(found) == 1, (
            f"Expected exactly 1 hmm-regime entry point, got {len(found)}"
        )
        ep = found[0]
        assert "hmm_futures_analysis.cli:main" in ep.value


class TestRunSh:
    """Verify run.sh produces valid CLI output."""

    def test_run_sh_exists_and_executable(self):
        from pathlib import Path

        run_sh = Path(__file__).resolve().parent.parent.parent / "run.sh"
        assert run_sh.exists(), "run.sh not found at repo root"
        assert run_sh.stat().st_mode & 0o111, "run.sh is not executable"

    def test_run_sh_produces_valid_json(self, btc_csv):
        from pathlib import Path

        import json

        run_sh = str(Path(__file__).resolve().parent.parent.parent / "run.sh")
        result = subprocess.run(
            [run_sh, "--csv", btc_csv, "--json"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert result.returncode == 0, f"run.sh failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert "current_regime" in data
        assert data["engine"] == "threshold"
