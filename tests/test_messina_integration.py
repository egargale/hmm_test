"""Integration tests for --messina CLI flag and OHLCV HMM path."""
import json

from tests.conftest import run_regime


class TestMessinaCLI:
    def test_messina_flag_json_output(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--messina")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "hmm" in data

    def test_messina_hmm_available(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--messina")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        hmm = data["hmm"]
        assert hmm["available"] is True, f"HMM not available: {hmm.get('reason')}"
        assert hmm.get("feature_mode") == "messina"

    def test_messina_hmm_has_regime_data(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--messina")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        hmm = data["hmm"]
        assert "regimes" in hmm
        assert "transition_matrix" in hmm
        assert "stationary_distribution" in hmm
        assert set(hmm["stationary_distribution"].keys()) == {"bear", "sideways", "bull"}

    def test_messina_does_not_break_threshold(self, btc_csv):
        """Threshold-based results should be unaffected by --messina."""
        result = run_regime("--csv", btc_csv, "--json", "--messina")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert -1.0 <= data["signal"] <= 1.0
        assert data["current_regime"]["name"] in ("bear", "sideways", "bull")

    def test_generic_mode_still_works(self, btc_csv):
        """Generic (non-messina) mode should still work."""
        result = run_regime("--csv", btc_csv, "--json")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        hmm = data["hmm"]
        assert hmm.get("feature_mode") == "generic"
