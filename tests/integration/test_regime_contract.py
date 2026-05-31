"""Contract validation tests for regime.py JSON output."""
import json

from tests.conftest import run_regime
import pytest

# Required top-level keys in the JSON contract (v0.2.0)
REQUIRED_KEYS = [
    "source", "engine", "dates",
    "current_regime", "next_state_probabilities",
    "signal", "transition_matrix", "persistence_diagonal",
    "stationary_distribution", "regime_counts", "walk_forward",
    "forecast", "engine_info", "framework", "disclaimer",
]


@pytest.mark.slow
class TestJSONContract:
    """Validate JSON output schema against the regime contract."""

    def test_all_required_keys_present(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        for key in REQUIRED_KEYS:
            assert key in data, f"Missing required key: {key}"

    def test_signal_is_float_in_range(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        assert isinstance(data["signal"], (int, float))
        assert -1.0 <= data["signal"] <= 1.0

    def test_transition_matrix_3x3(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        mat = data["transition_matrix"]
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

    def test_transition_matrix_rows_sum_to_one(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        for row in data["transition_matrix"]:
            assert abs(sum(row) - 1.0) < 0.01

    def test_stationary_distribution_sums_to_one(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        sd = data["stationary_distribution"]
        total = sd["bear"] + sd["sideways"] + sd["bull"]
        assert abs(total - 1.0) < 0.01

    def test_next_state_probabilities_sums_to_one(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        nsp = data["next_state_probabilities"]
        total = nsp["bear"] + nsp["sideways"] + nsp["bull"]
        assert abs(total - 1.0) < 0.01

    def test_engine_field(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        assert data["engine"] == "threshold"

    def test_walk_forward_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        wf = data["walk_forward"]
        expected = {"sharpe", "max_drawdown", "n_trades", "win_rate", "profit_factor", "total_return"}
        assert set(wf.keys()) == expected

    def test_engine_info_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        ei = data["engine_info"]
        assert "method" in ei
        assert "features" in ei
        assert "n_states" in ei

    def test_forecast_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        forecast = data["forecast"]
        assert "1_step" in forecast
        assert "5_step" in forecast
        assert "20_step" in forecast

    def test_regime_counts_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        rc = data["regime_counts"]
        assert set(rc.keys()) == {"bear", "sideways", "bull"}

    def test_dates_structure(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        assert "start" in data["dates"]
        assert "end" in data["dates"]

    def test_framework_string(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        assert "hmm_test" in data["framework"]

    def test_disclaimer_present(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        data = json.loads(result.stdout)
        assert len(data["disclaimer"]) > 0

    def test_engine_messina_csv_with_ohlcv_succeeds(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--engine", "messina")
        data = json.loads(result.stdout)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert data["engine"] == "messina"

    def test_engine_messina_csv_without_ohlcv_errors(self, tmp_path):
        lines = ["date,close"] + [f"2024-01-{d:02d},{100+d}" for d in range(1, 32)]
        close_only = tmp_path / "close_only.csv"
        close_only.write_text("\n".join(lines) + "\n")
        result = run_regime("--csv", str(close_only), "--json", "--engine", "messina")
        data = json.loads(result.stdout)
        assert "error" in data
        assert "OHLCV" in data["error"]


@pytest.mark.slow
class TestContractErrorHandling:
    """Test error handling in JSON mode."""

    def test_missing_file_returns_error_json(self):
        result = run_regime("--csv", "NONEXISTENT_FILE.csv", "--json")
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert "error" in data

    def test_missing_source_returns_nonzero(self):
        result = run_regime("--json")
        assert result.returncode != 0

    def test_small_dataset_json_valid(self, futures_csv):
        result = run_regime(
            "--csv", futures_csv, "--json", "--engine", "threshold"
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        # walk_forward may have null sharpe for small datasets
        wf_sharpe = data["walk_forward"]["sharpe"]
        assert wf_sharpe is None or isinstance(wf_sharpe, (int, float))
