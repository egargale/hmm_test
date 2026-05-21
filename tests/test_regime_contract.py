"""Contract validation tests for regime.py JSON output."""
import json

import pytest

from tests.conftest import run_regime

# Required top-level keys in the JSON contract
REQUIRED_KEYS = [
    "source", "rows", "date_start", "date_end", "params",
    "states", "current_regime", "next_state_probabilities",
    "signal", "transition_matrix", "persistence_diagonal",
    "stationary_distribution", "walk_forward", "hmm",
    "hmm_test_extras", "forecast", "framework", "disclaimer",
]


class TestJSONContract:
    """Validate JSON output schema against the regime contract."""

    def test_all_required_keys_present(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        for key in REQUIRED_KEYS:
            assert key in data, f"Missing required key: {key}"

    def test_signal_is_float_in_range(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        assert isinstance(data["signal"], (int, float))
        assert -1.0 <= data["signal"] <= 1.0

    def test_transition_matrix_3x3(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        mat = data["transition_matrix"]
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

    def test_transition_matrix_rows_sum_to_one(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        for row in data["transition_matrix"]:
            assert abs(sum(row) - 1.0) < 0.01

    def test_stationary_distribution_sums_to_one(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        sd = data["stationary_distribution"]
        total = sd["bear"] + sd["sideways"] + sd["bull"]
        assert abs(total - 1.0) < 0.01

    def test_next_state_probabilities_sums_to_one(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        nsp = data["next_state_probabilities"]
        total = nsp["bear"] + nsp["sideways"] + nsp["bull"]
        assert abs(total - 1.0) < 0.01

    def test_states_are_bear_sideways_bull(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        state_names = {s["name"] for s in data["states"]}
        assert state_names == {"bear", "sideways", "bull"}

    def test_walk_forward_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        wf = data["walk_forward"]
        assert "sharpe" in wf
        assert "max_drawdown" in wf
        assert "n_trades" in wf

    def test_hmm_block_present(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        assert "available" in data["hmm"]
        assert data["hmm"]["available"] is False  # --no-hmm

    def test_hmm_test_extras_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        extras = data["hmm_test_extras"]
        assert "n_states" in extras
        assert "method" in extras
        assert "data_points" in extras

    def test_forecast_keys(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        forecast = data["forecast"]
        assert "1_step" in forecast
        assert "5_step" in forecast
        assert "20_step" in forecast

    def test_params_reflect_args(self, btc_csv):
        result = run_regime(
            "--csv", btc_csv, "--json", "--no-hmm",
            "--window", "10", "--threshold", "0.03",
        )
        data = json.loads(result.stdout)
        assert data["params"]["window"] == 10
        assert data["params"]["threshold"] == 0.03

    def test_rows_positive(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        assert data["rows"] > 0

    def test_framework_string(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        assert "hmm_test" in data["framework"]

    def test_disclaimer_present(self, btc_csv):
        result = run_regime("--csv", btc_csv, "--json", "--no-hmm")
        data = json.loads(result.stdout)
        assert len(data["disclaimer"]) > 0


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
        result = run_regime("--csv", futures_csv, "--json", "--no-hmm")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        # walk_forward may have null sharpe for small datasets
        wf_sharpe = data["walk_forward"]["sharpe"]
        assert wf_sharpe is None or isinstance(wf_sharpe, (int, float))
