"""Real-data integration test suite across all test_data/ tickers.

Issue #99: Comprehensive real-data tests exercising all 5 engines
(threshold, hmm, messina, robust_hmm, fshmm) across diverse market profiles.

Tiered approach:
- Threshold engine: all loadable tickers (fast, deterministic)
- HMM engines: 6 representative tickers (slow, stochastic)

All tests marked @pytest.mark.slow. Run with: pytest -m slow
For parallel HMM: pytest -m slow -n auto
"""

import math
from pathlib import Path

import pytest

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import (
    FSHMMConfig,
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)
from hmm_futures_analysis.regime.pipeline import PipelineResult, run as pipeline_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = ROOT / "test_data"

HMM_SUBSET_TICKERS = ["SPY", "BTC", "MU", "XLU", "HYG", "AAPL"]

ENGINE_CONFIGS = {
    "threshold": ThresholdConfig,
    "hmm": HMMGenericConfig,
    "messina": HMMMMessinaConfig,
    "robust_hmm": RobustHMMConfig,
    "fshmm": FSHMMConfig,
}

HMM_ENGINES = {k: v for k, v in ENGINE_CONFIGS.items() if k != "threshold"}

REQUIRED_KEYS = [
    "source", "engine", "dates",
    "current_regime", "next_state_probabilities",
    "signal", "transition_matrix", "persistence_diagonal",
    "stationary_distribution", "regime_counts", "walk_forward",
    "forecast", "engine_info", "framework", "disclaimer", "verdict",
]

WF_REQUIRED_KEYS = {"sharpe", "max_drawdown", "n_trades", "win_rate",
                    "profit_factor", "total_return"}

VOLATILITY_TIERS = {
    "low": ["XLU", "LQD"],
    "mid": ["SPY", "KO", "AAPL", "BAC"],
    "high": ["BTC", "MU", "NVDA"],
    "credit": ["HYG", "LQD"],
}

# Parametrize constants
_HMM_PARAMS = [(t, e) for t in HMM_SUBSET_TICKERS for e in HMM_ENGINES]
_HMM_IDS = [f"{t}-{e}" for t, e in _HMM_PARAMS]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _safe_ticker_name(path: Path) -> str:
    return path.stem.replace(".", "_").replace("-", "_").replace("=", "_")


@pytest.fixture(scope="session")
def all_ticker_csvs() -> dict[str, Path]:
    """Discover all ticker CSVs in test_data/, excluding *_clean.csv."""
    result = {}
    for p in sorted(TEST_DATA_DIR.glob("*.csv")):
        if p.stem.endswith("_clean"):
            continue
        if p.parent != TEST_DATA_DIR:
            continue
        result[_safe_ticker_name(p)] = p
    return result


@pytest.fixture(scope="session")
def ticker_data(all_ticker_csvs):
    """Pre-load (prices, ohlcv, source) for all loadable tickers."""
    data = {}
    for name, path in all_ticker_csvs.items():
        try:
            data[name] = load_prices(csv=str(path))
        except Exception:
            pass  # Skip incompatible CSVs (e.g., SOXL multi-level headers)
    return data


@pytest.fixture(scope="session")
def threshold_results(ticker_data):
    """Cache threshold pipeline results for all tickers (fast: ~6s total)."""
    return {
        tname: pipeline_run(prices, source=tname, engine_config=ThresholdConfig())
        for tname, (prices, ohlcv, _) in ticker_data.items()
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_pipeline(tname, ename, ticker_data, **kwargs):
    """Run pipeline for one ticker×engine, return PipelineResult."""
    prices, ohlcv, _ = ticker_data[tname]
    ohlcv_arg = ohlcv if ename != "threshold" else None
    config = ENGINE_CONFIGS[ename]()
    return pipeline_run(prices, source=tname, engine_config=config,
                        ohlcv=ohlcv_arg, **kwargs)


def _assert_valid_regime_name(result, label=""):
    assert result.current_regime["name"] in {"bear", "sideways", "bull"}, \
        f"{label}: regime={result.current_regime['name']}"


def _assert_required_keys(result, label=""):
    for key in REQUIRED_KEYS:
        assert hasattr(result, key), f"{label}: missing {key}"


# ---------------------------------------------------------------------------
# Module 1: Ticker Discovery
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTickerDiscovery:

    def test_discovers_at_least_16(self, all_ticker_csvs):
        assert len(all_ticker_csvs) >= 16

    def test_excludes_clean_variants(self, all_ticker_csvs):
        assert all(not n.endswith("_clean") for n in all_ticker_csvs)

    def test_spy_btc_present(self, all_ticker_csvs):
        assert "SPY" in all_ticker_csvs
        assert "BTC" in all_ticker_csvs

    def test_0700_hk_safe_name(self, all_ticker_csvs):
        assert "0700_HK" in all_ticker_csvs

    def test_all_tickers_loadable(self, ticker_data, all_ticker_csvs):
        """Every discovered CSV either loads or is gracefully skipped."""
        loadable = set(ticker_data.keys())
        discovered = set(all_ticker_csvs.keys())
        # SOXL is known-unloadable; others should all load
        assert loadable.issuperset(discovered - {"SOXL"})


# ---------------------------------------------------------------------------
# Module 2: Smoke Tests — all tickers × all engines
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSmokeThreshold:

    def test_threshold_all_tickers_valid(self, threshold_results):
        for tname, result in threshold_results.items():
            assert isinstance(result, PipelineResult), f"{tname}"
            _assert_required_keys(result, tname)
            assert -1.0 <= result.signal <= 1.0
            _assert_valid_regime_name(result, tname)
            assert result.engine == "threshold"


@pytest.mark.slow
class TestSmokeHMM:

    @pytest.mark.parametrize("tname,ename", _HMM_PARAMS, ids=_HMM_IDS)
    def test_hmm_smoke(self, tname, ename, ticker_data):
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        result = _run_pipeline(tname, ename, ticker_data)
        label = f"{tname}-{ename}"
        assert isinstance(result, PipelineResult), label
        _assert_required_keys(result, label)
        assert -1.0 <= result.signal <= 1.0, f"{label}: signal={result.signal}"
        _assert_valid_regime_name(result, label)
        assert result.engine == ename


# ---------------------------------------------------------------------------
# Module 3: Walk-Forward Statistical Contracts
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestWalkForwardThreshold:

    def test_wf_contracts_all_tickers(self, threshold_results):
        for tname, result in threshold_results.items():
            wf = result.walk_forward
            # Keys present
            missing = WF_REQUIRED_KEYS - set(wf.keys())
            assert not missing, f"{tname}: missing {missing}"
            # Sharpe finite
            assert math.isfinite(wf["sharpe"]), f"{tname}: sharpe={wf['sharpe']}"
            # Drawdown ≤ 0 or NaN
            dd = wf["max_drawdown"]
            assert dd <= 0 or math.isnan(dd), f"{tname}: dd={dd}"
            # Trades > 0 (threshold always trades)
            assert wf["n_trades"] > 0, f"{tname}: n_trades=0"
            # Win rate in [0, 1]
            assert 0.0 <= wf["win_rate"] <= 1.0, f"{tname}: wr={wf['win_rate']}"
            # Profit factor > 0 when trades
            if wf["n_trades"] > 0:
                assert wf["profit_factor"] > 0, f"{tname}: pf={wf['profit_factor']}"
            # Total return finite
            assert math.isfinite(wf["total_return"]), f"{tname}: ret={wf['total_return']}"


@pytest.mark.slow
class TestWalkForwardHMM:

    @pytest.mark.parametrize("tname,ename", _HMM_PARAMS, ids=_HMM_IDS)
    def test_wf_contracts(self, tname, ename, ticker_data):
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        result = _run_pipeline(tname, ename, ticker_data)
        wf = result.walk_forward
        label = f"{tname}-{ename}"

        # Keys
        missing = WF_REQUIRED_KEYS - set(wf.keys())
        assert not missing, f"{label}: missing {missing}"
        # Sharpe finite or NaN
        s = wf["sharpe"]
        assert math.isfinite(s) or math.isnan(s), f"{label}: sharpe={s}"
        # Drawdown ≤ 0 or NaN
        dd = wf["max_drawdown"]
        assert dd <= 0 or math.isnan(dd), f"{label}: dd={dd}"
        # n_trades ≥ 0; zero requires degenerate flag
        nt = wf["n_trades"]
        assert nt >= 0, f"{label}: n_trades={nt}"
        if nt == 0:
            info = result.engine_info
            assert info.get("degenerate_fit") or info.get("auto_recovery"), \
                f"{label}: zero trades without degenerate flag"
        # Win rate
        wr = wf["win_rate"]
        if not math.isnan(wr):
            assert 0.0 <= wr <= 1.0, f"{label}: wr={wr}"
        # Total return finite or NaN
        ret = wf["total_return"]
        assert math.isfinite(ret) or math.isnan(ret), f"{label}: ret={ret}"


# ---------------------------------------------------------------------------
# Module 4: Regime Distribution Sanity
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRegimeDistributionThreshold:

    def test_regime_sanity_all_tickers(self, threshold_results):
        for tname, result in threshold_results.items():
            counts = result.regime_counts
            # Exactly 3 regimes
            assert set(counts.keys()) == {"bear", "sideways", "bull"}, \
                f"{tname}: {set(counts.keys())}"
            # All populated
            for regime, count in counts.items():
                assert count > 0, f"{tname}: {regime}=0"
            # At least 2 populated (trivially true if all 3 > 0)
            assert sum(1 for v in counts.values() if v > 0) >= 2


@pytest.mark.slow
class TestRegimeDistributionHMM:

    @pytest.mark.parametrize("tname,ename", _HMM_PARAMS, ids=_HMM_IDS)
    def test_regime_sanity(self, tname, ename, ticker_data):
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        result = _run_pipeline(tname, ename, ticker_data)
        label = f"{tname}-{ename}"
        n_states = result.engine_info.get("n_states", 3)
        counts = result.regime_counts

        # regime_counts always has 3 keys (bear, sideways, bull) even for 2-state
        # degenerate fits. n_states may be 2 for degenerate auto-recovery.
        assert len(counts) >= n_states, \
            f"{label}: {len(counts)} regimes < n_states={n_states}"
        # At least 2 populated
        assert sum(1 for v in counts.values() if v > 0) >= 2, \
            f"{label}: only {sum(1 for v in counts.values() if v > 0)} populated"


# ---------------------------------------------------------------------------
# Module 5: Transition Matrix Contracts
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTransitionMatrixThreshold:

    def test_transmat_contracts_all_tickers(self, threshold_results):
        for tname, result in threshold_results.items():
            tm = result.transition_matrix
            # 3×3
            assert len(tm) == 3, f"{tname}: {len(tm)} rows"
            assert all(len(r) == 3 for r in tm), f"{tname}: non-3 cols"
            for i, row in enumerate(tm):
                # Row sums to 1.0 ± 0.01
                assert abs(sum(row) - 1.0) <= 0.01, f"{tname}: row {i}={sum(row)}"
                # Diagonal in (0, 1]
                assert 0 < row[i] <= 1.0, f"{tname}: diag[{i}]={row[i]}"
                # Off-diagonal ≥ 0
                for j, v in enumerate(row):
                    if i != j:
                        assert v >= 0, f"{tname}: [{i}][{j}]={v}"


@pytest.mark.slow
class TestTransitionMatrixHMM:

    @pytest.mark.parametrize("tname,ename", _HMM_PARAMS, ids=_HMM_IDS)
    def test_transmat_contracts(self, tname, ename, ticker_data):
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        result = _run_pipeline(tname, ename, ticker_data)
        label = f"{tname}-{ename}"
        # Pipeline always outputs 3×3 transition matrix (bear, sideways, bull)
        # even for 2-state degenerate fits.
        tm = result.transition_matrix
        assert len(tm) == 3, f"{label}: {len(tm)} rows"
        assert all(len(r) == 3 for r in tm), f"{label}: non-3 cols"
        for i, row in enumerate(tm):
            assert abs(sum(row) - 1.0) <= 0.01, f"{label}: row {i}={sum(row)}"
            assert 0 < row[i] <= 1.0, f"{label}: diag[{i}]={row[i]}"
            for j, v in enumerate(row):
                if i != j:
                    assert v >= 0, f"{label}: [{i}][{j}]={v}"


# ---------------------------------------------------------------------------
# Module 6: Duration Forecast Contracts
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDurationForecastThreshold:

    def test_duration_contracts_all_tickers(self, ticker_data):
        for tname, (prices, ohlcv, _) in ticker_data.items():
            result = pipeline_run(prices, source=tname,
                                  engine_config=ThresholdConfig(),
                                  duration_forecast=True)
            df = result.duration_forecast
            if df is None:
                continue
            assert df.get("expected_remaining_days", 0) > 0, \
                f"{tname}: remaining={df.get('expected_remaining_days')}"
            assert df.get("weibull_shape", 0) > 0, \
                f"{tname}: shape={df.get('weibull_shape')}"
            assert df.get("weibull_scale", 0) > 0, \
                f"{tname}: scale={df.get('weibull_scale')}"


@pytest.mark.slow
class TestDurationForecastHMM:

    @pytest.mark.parametrize("tname,ename", _HMM_PARAMS, ids=_HMM_IDS)
    def test_duration_contracts(self, tname, ename, ticker_data):
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        result = _run_pipeline(tname, ename, ticker_data, duration_forecast=True)
        df = result.duration_forecast
        if df is None:
            return
        label = f"{tname}-{ename}"
        assert df.get("expected_remaining_days", 0) > 0, \
            f"{label}: remaining={df.get('expected_remaining_days')}"
        assert df.get("weibull_shape", 0) > 0, \
            f"{label}: shape={df.get('weibull_shape')}"
        assert df.get("weibull_scale", 0) > 0, \
            f"{label}: scale={df.get('weibull_scale')}"


# ---------------------------------------------------------------------------
# Module 7: Engine Output Shape Consistency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEngineOutputShapeConsistency:

    @pytest.mark.parametrize("tname", HMM_SUBSET_TICKERS)
    def test_keys_identical_across_engines(self, tname, ticker_data):
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        prices, ohlcv, _ = ticker_data[tname]
        key_sets = {}
        for ename, ecls in ENGINE_CONFIGS.items():
            ohlcv_arg = ohlcv if ename != "threshold" else None
            r = pipeline_run(prices, source=tname, engine_config=ecls(),
                             ohlcv=ohlcv_arg)
            key_sets[ename] = set(r._asdict().keys())
            # Verdict is a dict
            assert isinstance(r.verdict, dict), f"{tname}-{ename}: verdict type"

        ref = key_sets["threshold"]
        for ename, keys in key_sets.items():
            assert keys == ref, \
                f"{tname}: {ename} differs: {keys.symmetric_difference(ref)}"


# ---------------------------------------------------------------------------
# Module 8: Volatility-Stratified Behavioral Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestVolatilityStratified:

    def test_threshold_high_vol_positive_trades_and_finite_sharpe(
        self, threshold_results,
    ):
        """Threshold: positive trades + finite Sharpe on high-vol tickers."""
        for tname in VOLATILITY_TIERS["high"]:
            if tname not in threshold_results:
                continue
            result = threshold_results[tname]
            assert result.walk_forward["n_trades"] > 0, \
                f"{tname}: high-vol zero trades"
            assert math.isfinite(result.walk_forward["sharpe"]), \
                f"{tname}: Sharpe not finite"

    @pytest.mark.parametrize("tname", VOLATILITY_TIERS["low"])
    def test_hmm_survives_low_vol(self, tname, ticker_data):
        """HMM engines complete without error on low-vol tickers."""
        if tname not in ticker_data:
            pytest.skip(f"{tname} not available")
        prices, ohlcv, _ = ticker_data[tname]
        for ename, ecls in HMM_ENGINES.items():
            result = pipeline_run(prices, source=tname,
                                  engine_config=ecls(), ohlcv=ohlcv)
            assert isinstance(result, PipelineResult), \
                f"{tname}-{ename}: not PipelineResult"
            # engine_info has n_states
            assert "n_states" in result.engine_info, \
                f"{tname}-{ename}: missing n_states"
