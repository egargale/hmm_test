"""Multi-ticker evaluation harness.

Runs one or more engines against multiple tickers or CSV files and
produces a comparison table (markdown to stderr) or JSON (to stdout).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .data_processing.csv_auto_detect import load_prices
from .presenter import format_eval as _format_eval
from .regime.engine_configs import (
    FSHMMConfig,
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)
from .regime.pipeline import run as pipeline_run

ALL_ENGINES = ("threshold", "messina", "hmm", "robust_hmm", "fshmm")

_ENGINE_CONFIG_DEFAULTS: dict[str, type] = {
    "threshold": ThresholdConfig,
    "messina": HMMMMessinaConfig,
    "hmm": HMMGenericConfig,
    "robust_hmm": RobustHMMConfig,
    "fshmm": FSHMMConfig,
}


# Re-export for backwards compatibility — format_table has moved to presenter.format_eval.
def format_table(results: list[dict[str, Any]]) -> str:
    """Format results as a markdown table.

    .. deprecated::
       Use :func:`presenter.format_eval(results, fmt="table")` instead.
    """
    return _format_eval(results, fmt="table")


def _make_config(engine_name: str) -> object:
    """Create a default engine config for the given engine name."""
    return _ENGINE_CONFIG_DEFAULTS[engine_name]()


def _extract_summary(result: Any, wall_seconds: float) -> dict[str, Any]:
    """Extract evaluation-relevant fields from a PipelineResult."""
    wf = result.walk_forward
    return {
        "ticker": result.source,
        "engine": result.engine,
        "regime": result.current_regime["name"],
        "signal": round(result.signal, 4),
        "sharpe": wf["sharpe"],
        "max_drawdown": wf["max_drawdown"],
        "n_trades": wf["n_trades"],
        "win_rate": wf["win_rate"],
        "profit_factor": wf["profit_factor"],
        "total_return": wf["total_return"],
        "wall_seconds": round(wall_seconds, 3),
    }


def run_eval_csv(
    csv_dir: str,
    engines: tuple[str, ...] = ALL_ENGINES,
    min_train: int = 252,
) -> list[dict[str, Any]]:
    """Run evaluation from a directory of CSV files.

    Each file's stem (without extension) is used as the ticker name.
    Returns a list of summary dicts.
    """
    csv_path = Path(csv_dir)
    if not csv_path.is_dir():
        raise ValueError(f"Not a directory: {csv_dir}")

    csv_files = sorted(csv_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    results: list[dict[str, Any]] = []
    for csv_file in tqdm(csv_files, desc="CSVs", position=0, leave=True):
        ticker = csv_file.stem
        prices, ohlcv, source = load_prices(csv=str(csv_file))
        for engine_name in tqdm(engines, desc=f"  {ticker}", position=1, leave=False):
            config = _make_config(engine_name)
            t0 = time.monotonic()
            output = pipeline_run(
                prices=prices,
                source=ticker,
                engine_config=config,
                min_train=min_train,
                ohlcv=ohlcv,
            )
            elapsed = time.monotonic() - t0
            results.append(_extract_summary(output, elapsed))

    return results


def run_eval_tickers(
    tickers: tuple[str, ...],
    cache_dir: str | None = None,
    refresh: bool = False,
    no_cache: bool = False,
    engines: tuple[str, ...] = ALL_ENGINES,
    min_train: int = 252,
) -> list[dict[str, Any]]:
    """Run evaluation from yfinance tickers.

    Fetches data, using the disk cache when available, then runs all
    requested engines.
    """
    results: list[dict[str, Any]] = []
    for ticker in tqdm(tickers, desc="Tickers", position=0, leave=True):
        prices, ohlcv, _source = load_prices(
            ticker=ticker,
            cache_dir=cache_dir,
            refresh=refresh,
            no_cache=no_cache,
        )

        for engine_name in tqdm(engines, desc=f"  {ticker}", position=1, leave=False):
            config = _make_config(engine_name)
            t0 = time.monotonic()
            output = pipeline_run(
                prices=prices,
                source=ticker,
                engine_config=config,
                min_train=min_train,
                ohlcv=ohlcv,
            )
            elapsed = time.monotonic() - t0
            results.append(_extract_summary(output, elapsed))

    return results


# Statistical metrics (issue #114, Slice 3) ---------------------------------
#
# The metric *definitions* are pinned by the Slice 2 harness docstring
# (tests/integration/test_statistical_evaluation.py) per ADR-0027.  The
# implementations below are the single code path shared by that harness and
# the --eval-statistical CLI surface (ADR-0028), so the two can never drift.

_CATEGORIES = ("bear", "sideways", "bull")


def fleiss_components(
    counts: list[list[int]],
) -> tuple[float, float, float]:
    """Return ``(p_bar, p_e, kappa)`` for a subjects x categories matrix.

    ``counts[i][j]`` = number of raters assigning subject *i* to category *j*.
    Raises ``ValueError`` when kappa is undefined (fewer than two raters per
    subject, or chance agreement exactly 1) — the Slice 2 harness contract.
    """
    counts_arr = np.asarray(counts, dtype=float)
    if counts_arr.ndim != 2 or counts_arr.size == 0:
        raise ValueError("counts must be a non-empty 2D matrix")
    n_raters = counts_arr.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        raise ValueError("each subject must be rated by the same number of raters")
    n = n_raters[0]
    if n < 2:
        raise ValueError("kappa needs at least two raters per subject")

    # Per-subject agreement: (sum_j n_ij^2 - n) / (n*(n-1))
    p_i = (counts_arr**2).sum(axis=1) - n
    p_i = p_i / (n * (n - 1))
    p_bar = float(p_i.mean())

    # Category prevalence across all ratings
    p_cat = counts_arr.sum(axis=0) / (counts_arr.sum())
    p_e = float((p_cat**2).sum())

    if p_e == 1.0:
        raise ValueError("kappa undefined: chance agreement is exactly 1")
    kappa = (p_bar - p_e) / (1.0 - p_e)
    return p_bar, p_e, float(kappa)


def fleiss_kappa(counts_matrix: list[list[int]]) -> float:
    """Fleiss' kappa for a subjects x categories vote-count matrix.

    The Slice 2 panel is 5 tickers x 5 engine votes over 3 categories
    (bear/sideways/bull).  Raises ``ValueError`` when kappa is undefined.
    """
    return fleiss_components(counts_matrix)[2]


def buy_and_hold_stats(
    prices: np.ndarray, min_train: int = 252
) -> dict[str, float]:
    """Buy-and-hold Sharpe / return / max drawdown on the engine window.

    Metrics use the engine's own window (``prices[min_train:]``): return is
    measured vs the first point of that window; rf=0 in all Sharpe ratios;
    vol uses sample std (ddof=1); annualisation is ``sqrt(252)``; max
    drawdown is running peak-to-trough on the window.  (ADR-0027.)
    """
    px = np.asarray(prices, dtype=float)[min_train:]
    if px.size < 2:
        raise ValueError(
            "buy_and_hold_stats needs >= 2 prices after "
            f"min_train={min_train}, got {px.size}"
        )
    rets = np.diff(px) / px[:-1]
    return {
        "sharpe": float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)),
        "total_return": float(px[-1] / px[0] - 1),
        "max_drawdown": float(np.min(px / np.maximum.accumulate(px) - 1)),
    }


def _statistical_fixture_csv(ticker: str) -> Path | None:
    """Resolve *ticker* to a committed ``test_data/<TICKER>.csv`` fixture.

    Statistical mode is defined on the Slice 2 harness data, so a committed
    fixture wins over the yfinance path.  Search order: ``./test_data``
    (working directory, as the harness maps it on a clean clone), then the
    repository root next to this package.  ``None`` means "no fixture" and
    the caller falls back to the ticker path (ADR-0004).
    """
    if Path(ticker).name != ticker:
        return None
    repo_root = Path(__file__).resolve().parent.parent
    for root in (Path.cwd() / "test_data", repo_root / "test_data"):
        candidate = root / f"{ticker}.csv"
        if candidate.is_file():
            return candidate
    return None


def run_eval_statistical(
    tickers: tuple[str, ...],
    engines: tuple[str, ...] = ALL_ENGINES,
    min_train: int = 252,
    cache_dir: str | None = None,
    refresh: bool = False,
    no_cache: bool = False,
) -> list[dict[str, Any]]:
    """Run the statistical evaluation panel (issue #114, Slice 3).

    Same row shape as :func:`run_eval_tickers`, plus two fields per row:

    - ``fleiss_kappa`` — single panel-wide float (identical on every row),
      cross-engine agreement over the ``current_regime`` votes; ``None``
      when undefined (fewer than two engines, or chance agreement 1).
    - ``bnh_delta`` — engine walk-forward Sharpe minus the buy-and-hold
      Sharpe on the engine window ``prices[min_train:]`` (ADR-0027).

    ``pipeline_run`` is invoked with the CLI single-run filter defaults
    (``dwell_bars="auto"``, ``hysteresis_delta="auto"``) so the row values,
    votes and ``bnh_delta`` match the Slice 2 harness cells (ADR-0028).

    Each ticker resolves against committed ``test_data/<TICKER>.csv``
    fixtures first, falling back to the yfinance ticker path.
    """
    tickers = tuple(tickers)
    engines = tuple(engines)
    results: list[dict[str, Any]] = []
    for ticker in tqdm(tickers, desc="Tickers", position=0, leave=True):
        fixture = _statistical_fixture_csv(ticker)
        if fixture is not None:
            prices, ohlcv, _source = load_prices(csv=str(fixture))
        else:
            prices, ohlcv, _source = load_prices(
                ticker=ticker,
                cache_dir=cache_dir,
                refresh=refresh,
                no_cache=no_cache,
            )
        bh = buy_and_hold_stats(prices.to_numpy(), min_train=min_train)

        for engine_name in tqdm(engines, desc=f"  {ticker}", position=1, leave=False):
            config = _make_config(engine_name)
            t0 = time.monotonic()
            output = pipeline_run(
                prices=prices,
                source=ticker,
                engine_config=config,
                min_train=min_train,
                ohlcv=ohlcv,
                dwell_bars="auto",
                hysteresis_delta="auto",
            )
            elapsed = time.monotonic() - t0
            row = _extract_summary(output, elapsed)
            row["bnh_delta"] = row["sharpe"] - bh["sharpe"]
            results.append(row)

    # Panel-wide Fleiss' kappa over the engine votes (undefined -> None).
    counts = [[0 for _ in _CATEGORIES] for _ in tickers]
    ticker_row = {ticker: i for i, ticker in enumerate(tickers)}
    for row in results:
        counts[ticker_row[row["ticker"]]][_CATEGORIES.index(row["regime"])] += 1
    try:
        kappa: float | None = fleiss_kappa(counts)
    except ValueError:
        kappa = None
    for row in results:
        row["fleiss_kappa"] = kappa

    return results
