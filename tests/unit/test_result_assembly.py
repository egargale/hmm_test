"""Tests for _result_assembly module — result construction extracted from pipeline.

These tests verify that _assemble_result() produces a correct PipelineResult
from its constituent parts, without any pipeline/runner side effects.
"""

import numpy as np
import pytest

from hmm_futures_analysis.regime._result_assembly import (
    PipelineResult,
    _assemble_result,
)


def _make_markov_stats(**overrides):
    """Create a minimal MarkovStats-like dict for testing."""
    base = {
        "transmat": np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]),
        "stationary": np.array([0.2, 0.5, 0.3]),
        "persistence": {"bear": 0.7, "sideways": 0.8, "bull": 0.7},
        "signal": 0.35,
        "current_regime": 2,
        "current_probs": np.array([0.1, 0.2, 0.7]),
        "regime_counts": {"bear": 50, "sideways": 120, "bull": 80},
        "dates": {"start": "2024-01-01", "end": "2024-12-31"},
    }
    base.update(overrides)
    return type("_MarkovStats", (), base)()


def _make_classify_out(**overrides):
    """Create a minimal ClassifyOutput-like object for testing."""
    base = {
        "regimes": np.array([0, 1, 1, 2, 2, 2]),
        "posteriors": None,
        "n_states": 3,
        "engine_info": None,
        "reverse_classify": False,
    }
    base.update(overrides)
    return type("_ClassifyOut", (), base)()


class TestAssembleResultBasic:
    """Tracer bullet: _assemble_result returns a PipelineResult with all fields."""

    def test_returns_pipeline_result_type(self):
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=_make_markov_stats(),
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert isinstance(result, PipelineResult)

    def test_preserves_source_and_engine(self):
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=_make_markov_stats(),
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert result.source == "SPY"
        assert result.engine == "threshold"

    def test_constructs_dates_from_markov(self):
        markov = _make_markov_stats()
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=markov,
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert result.dates == {"start": "2024-01-01", "end": "2024-12-31"}

    def test_current_regime_named_correctly(self):
        markov = _make_markov_stats(current_regime=2)
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=markov,
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert result.current_regime == {"name": "bull", "index": 2}

    def test_verdict_computed_for_bull_regime(self):
        markov = _make_markov_stats(current_regime=2, signal=0.7)
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=markov,
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert result.verdict["verdict"] == "bullish"
        assert result.verdict["confidence"] == 0.7

    def test_optional_fields_default_to_none(self):
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=_make_markov_stats(),
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert result.duration_forecast is None
        assert result.timing is None

    def test_duration_forecast_forwarded(self):
        df = {"weibull_scale": 50.0, "weibull_shape": 1.5, "days_in_regime": 10}
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=_make_markov_stats(),
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing={"total_wall_seconds": 0.1, "phases": {}},
            duration_forecast_result=df,
            regime_transitions=[],
        )
        assert result.duration_forecast == df
        assert result.timing is not None

    def test_transition_matrix_as_list(self):
        markov = _make_markov_stats()
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=markov,
            walk_forward={
                "sharpe": 1.5,
                "max_drawdown": -0.12,
                "n_trades": 10,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_return": 0.25,
            },
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        assert isinstance(result.transition_matrix, list)
        assert len(result.transition_matrix) == 3
        for row in result.transition_matrix:
            assert isinstance(row, list)
            assert len(row) == 3


class TestForecastAlgebraIssue112:
    """forecast.1_step must equal T[current_regime] (issue #112).

    current_probs is the one-hot e_current, so forecast_n_steps applies
    exactly one matrix multiplication: forecast.1_step and
    next_state_probabilities both carry the current-regime row of T.
    The high-persistence matrix (the case where the old row-as-input bug
    was masked, T^2 ~= T) is covered explicitly.
    """

    WALK_FORWARD = {
        "sharpe": 1.5,
        "max_drawdown": -0.12,
        "n_trades": 10,
        "win_rate": 0.6,
        "profit_factor": 1.8,
        "total_return": 0.25,
    }

    @pytest.mark.parametrize(
        "transmat",
        [
            np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]),
            np.array(
                [
                    [0.999, 0.0005, 0.0005],
                    [0.001, 0.998, 0.001],
                    [0.0005, 0.0005, 0.999],
                ]
            ),
        ],
        ids=["standard", "high_persistence"],
    )
    def test_forecast_1_step_equals_transmat_row(self, transmat):
        i = 1
        markov = _make_markov_stats(
            transmat=transmat,
            current_regime=i,
            current_probs=np.eye(3)[i],
            signal=0.35,
        )
        result = _assemble_result(
            source="SPY",
            engine_name="threshold",
            markov=markov,
            walk_forward=dict(self.WALK_FORWARD),
            engine_info={"method": "threshold", "features": "returns", "n_states": 3},
            timing=None,
            duration_forecast_result=None,
            regime_transitions=[],
        )
        keys = ("bear", "sideways", "bull")
        forecast_vec = np.array([result.forecast["1_step"][k] for k in keys])
        nsp_vec = np.array([result.next_state_probabilities[k] for k in keys])
        np.testing.assert_allclose(forecast_vec, transmat[i], rtol=0, atol=1e-12)
        np.testing.assert_allclose(nsp_vec, transmat[i], rtol=0, atol=1e-12)
        # SKILL.md contract: the two JSON fields carry the same distribution
        np.testing.assert_array_equal(forecast_vec, nsp_vec)
        # wire signal untouched by the one-hot change
        assert result.signal == 0.35
        # pre-fix witness: the row-as-input value produced T^2 here
        buggy = transmat[i] @ transmat
        assert not np.allclose(forecast_vec, buggy, rtol=0, atol=1e-9)
