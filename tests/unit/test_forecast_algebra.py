"""End-to-end forecast algebra (issue #112, slice 1 acceptance).

With ``MarkovStats.current_probs`` as the one-hot ``e_current``, the JSON
``forecast.1_step`` field and ``next_state_probabilities`` must both equal the
row of the transition matrix ``T[current_regime, :]`` to floating-point
tolerance — on every committed test-data ticker with the threshold engine,
and on a synthetic HMM run (the high-persistence masking case).

Everything is offline: committed CSVs under ``test_data/`` and synthetic
frames only.  No network.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig, ThresholdConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The threshold panel named in the issue/repro (all five test-data tickers).
PANEL = ["SPY", "BTC", "NVDA", "KO", "XLE"]

_PROB_KEYS = ("bear", "sideways", "bull")


def _vec(prob_dict) -> np.ndarray:
    return np.array([prob_dict[k] for k in _PROB_KEYS])


def _assert_forecast_is_transmat_row(result, atol: float = 1e-9) -> None:
    i = result.current_regime["index"]
    transmat = np.array(result.transition_matrix, dtype=float)
    forecast_1 = _vec(result.forecast["1_step"])
    nsp = _vec(result.next_state_probabilities)
    # forecast.1_step == T[current_regime]
    np.testing.assert_allclose(forecast_1, transmat[i], rtol=0, atol=atol)
    # next_state_probabilities unchanged: still the current-regime row
    np.testing.assert_allclose(nsp, transmat[i], rtol=0, atol=atol)
    # SKILL.md JSON contract: the two fields are the same distribution
    np.testing.assert_allclose(forecast_1, nsp, rtol=0, atol=0)
    # sanity: still a probability distribution
    np.testing.assert_allclose(forecast_1.sum(), 1.0, rtol=0, atol=1e-9)


class TestThresholdPanel:
    """Every test-data ticker, threshold engine, in-process pipeline.run."""

    def test_threshold_forecast_1_step_equals_transmat_row(self):
        for ticker in PANEL:
            prices = load_from_csv(str(REPO_ROOT / "test_data" / f"{ticker}.csv"))
            result = pipeline_run(
                prices,
                source=ticker,
                engine_config=ThresholdConfig(),
                profile=False,
            )
            _assert_forecast_is_transmat_row(result)


def _synthetic_inputs(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    prices = pd.Series(close, index=idx, name="close")
    ohlcv = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000, 100_000, n).astype(float),
        },
        index=idx,
    )
    return prices, ohlcv


class TestHmmSynthetic:
    """HMM engine on synthetic inputs (no HMM fitting fixtures needed)."""

    def test_hmm_forecast_1_step_equals_transmat_row(self):
        prices, ohlcv = _synthetic_inputs()
        result = pipeline_run(
            prices,
            source="SYNTHETIC",
            engine_config=HMMGenericConfig(n_states=3),
            ohlcv=ohlcv,
            profile=False,
        )
        _assert_forecast_is_transmat_row(result)
