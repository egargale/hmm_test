"""Integration tests for --engine messina and --engine hmm CLI paths."""
import json

from tests.conftest import run_regime


class TestMessinaCLI:
    def test_engine_messina_csv_without_ohlcv_errors(self, btc_csv):
        """--csv with --engine messina errors without OHLCV columns."""
        result = run_regime("--csv", btc_csv, "--json", "--engine", "messina")
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert "OHLCV" in data.get("error", "")

    def test_engine_hmm_csv_without_ohlcv_errors(self, btc_csv):
        """--csv with --engine hmm errors without OHLCV columns."""
        result = run_regime("--csv", btc_csv, "--json", "--engine", "hmm")
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert "OHLCV" in data.get("error", "")

    def test_threshold_is_default(self, btc_csv):
        """Default engine is threshold (no --engine flag)."""
        result = run_regime("--csv", btc_csv, "--json")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine"] == "threshold"
