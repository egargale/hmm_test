"""Integration tests for --engine messina and --engine hmm CLI paths."""
import json

from tests.conftest import run_regime


class TestMessinaCLI:
    def test_engine_messina_csv_with_ohlcv_succeeds(self, btc_csv):
        """BTC.csv has OHLCV columns, so messina should work."""
        result = run_regime("--csv", btc_csv, "--json", "--engine", "messina")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine"] == "messina"

    def test_engine_hmm_csv_with_ohlcv_succeeds(self, btc_csv):
        """BTC.csv has OHLCV columns, so hmm should work."""
        result = run_regime("--csv", btc_csv, "--json", "--engine", "hmm")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine"] == "hmm"

    def test_threshold_is_default(self, btc_csv):
        """Default engine is threshold (no --engine flag)."""
        result = run_regime("--csv", btc_csv, "--json")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine"] == "threshold"
