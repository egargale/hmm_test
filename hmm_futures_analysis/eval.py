"""Multi-ticker evaluation harness.

Runs one or more engines against multiple tickers or CSV files and
produces a comparison table (markdown to stderr) or JSON (to stdout).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .data_processing.csv_auto_detect import load_prices
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


def _make_config(engine_name: str) -> object:
    """Create a default engine config for the given engine name."""
    return _ENGINE_CONFIG_DEFAULTS[engine_name]()


def _save_ticker_csv(ticker: str, output_dir: Path) -> Path:
    """Fetch a ticker via yfinance and save to CSV. Returns the CSV path."""
    import yfinance as yf

    data = yf.download(ticker, period="10y", progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    safe_name = ticker.replace(".", "_").replace("=", "_").replace("-", "_")
    csv_path = output_dir / f"{safe_name}.csv"
    data.to_csv(csv_path)
    return csv_path


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
    for csv_file in csv_files:
        ticker = csv_file.stem
        prices, ohlcv, source = load_prices(csv=str(csv_file))
        for engine_name in engines:
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
    csv_cache_dir: str | None = None,
    engines: tuple[str, ...] = ALL_ENGINES,
    min_train: int = 252,
) -> list[dict[str, Any]]:
    """Run evaluation from yfinance tickers.

    Fetches data, optionally saves to CSV for reproducibility, then
    runs all requested engines.
    """
    cache_dir = Path(csv_cache_dir) if csv_cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for ticker in tickers:
        if cache_dir:
            csv_path = _save_ticker_csv(ticker, cache_dir)
            prices, ohlcv, _source = load_prices(csv=str(csv_path))
        else:
            prices, ohlcv, _source = load_prices(ticker=ticker)

        for engine_name in engines:
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


def format_table(results: list[dict[str, Any]]) -> str:
    """Format results as a markdown table."""
    if not results:
        return "No results."

    headers = [
        "ticker", "engine", "regime", "signal", "sharpe", "max_dd",
        "trades", "win_rate", "pf", "total_ret", "wall_s",
    ]

    def fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            if abs(val) < 100:
                return f"{val:.3f}"
            return f"{val:.1f}"
        return str(val)

    rows: list[list[str]] = []
    for r in results:
        rows.append([
            r["ticker"],
            r["engine"],
            r["regime"],
            fmt(r["signal"]),
            fmt(r["sharpe"]),
            fmt(r["max_drawdown"]),
            str(r["n_trades"]),
            fmt(r["win_rate"]),
            fmt(r["profit_factor"]),
            fmt(r["total_return"]),
            fmt(r["wall_seconds"]),
        ])

    # Column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    def separator() -> str:
        return "  ".join("-" * w for w in widths)

    lines = [
        fmt_row(headers),
        separator(),
    ]
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)
