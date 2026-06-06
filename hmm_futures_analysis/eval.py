"""Multi-ticker evaluation harness.

Runs one or more engines against multiple tickers or CSV files and
produces a comparison table (markdown to stderr) or JSON (to stdout).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

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



