#!/usr/bin/env python3
"""Regime detection entry point for the hmm-regime-detection skill.

CLI for detecting market regimes (Bull/Bear/Sideways) using one of three
independent engines: threshold (fast, close-only), messina (HMM + 12
Messina features), or hmm (HMM + ~44 generic features).

Usage:
    python scripts/cli.py --csv BTC.csv --json
    python scripts/cli.py --ticker ES=F --json --engine hmm
    python scripts/cli.py --csv data.csv --engine threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure scripts/ directory is on sys.path so that sibling packages resolve
# when this file is executed directly (e.g. `python scripts/regime.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data_processing.csv_auto_detect import load_from_csv
from regime.pipeline import run as pipeline_run

_STATE_NAMES = ("bear", "sideways", "bull")
_FRAMEWORK_VERSION = "hmm_test v0.2.0"


def _print_terminal(output: dict) -> None:
    """Pretty-print regime analysis results to stderr."""
    width = 54
    sep = "─" * width

    def header(title: str) -> None:
        print(f"\n{sep}", file=sys.stderr)
        print(f"  {title}", file=sys.stderr)
        print(sep, file=sys.stderr)

    sr = output
    header("REGIME DETECTION")
    print(f"  Source      : {sr['source']}", file=sys.stderr)
    print(f"  Engine      : {sr['engine']}", file=sys.stderr)
    print(f"  Date range  : {sr['dates']['start']} → {sr['dates']['end']}", file=sys.stderr)

    ei = sr["engine_info"]
    print(f"  Method      : {ei['method']}", file=sys.stderr)
    print(f"  Features    : {ei['features']}", file=sys.stderr)
    if "caveat" in ei:
        print(f"  Caveat      : {ei['caveat']}", file=sys.stderr)

    header("CURRENT REGIME")
    cr = sr["current_regime"]
    print(f"  Regime      : {cr['name'].upper()} (index {cr['index']})", file=sys.stderr)
    print(f"  Signal      : {sr['signal']:+.4f}", file=sys.stderr)

    header("REGIME DISTRIBUTION")
    rc = sr["regime_counts"]
    for name in _STATE_NAMES:
        print(f"  {name.capitalize():<12s}: {rc.get(name, 0):>6d}", file=sys.stderr)

    header("NEXT-STATE PROBABILITIES")
    for name in _STATE_NAMES:
        prob = sr["next_state_probabilities"][name]
        bar = "█" * int(prob * 30)
        print(f"  {name.capitalize():<12s}: {prob:.3f}  {bar}", file=sys.stderr)

    header("PERSISTENCE")
    for name in _STATE_NAMES:
        p = sr["persistence_diagonal"][name]
        print(f"  {name.capitalize():<12s}: {p:.3f}", file=sys.stderr)

    header("TRANSITION MATRIX")
    names = [s.capitalize() for s in _STATE_NAMES]
    print(f"  {'':>10s}  {names[0]:>8s}  {names[1]:>8s}  {names[2]:>8s}", file=sys.stderr)
    for i, row in enumerate(sr["transition_matrix"]):
        cells = "  ".join(f"{v:8.3f}" for v in row)
        print(f"  {names[i]:>10s}  {cells}", file=sys.stderr)

    header("STATIONARY DISTRIBUTION")
    for name in _STATE_NAMES:
        prob = sr["stationary_distribution"][name]
        print(f"  {name.capitalize():<12s}: {prob:.3f}", file=sys.stderr)

    header("FORECAST (n-step)")
    for step_key in ("1_step", "5_step", "20_step"):
        f = sr["forecast"][step_key]
        print(f"  {step_key.replace('_', ' ').capitalize()}:", file=sys.stderr)
        for name in _STATE_NAMES:
            print(f"    {name.capitalize():<10s}: {f[name]:.3f}", file=sys.stderr)

    header("WALK-FORWARD BACKTEST")
    wf = sr["walk_forward"]
    sharpe_str = f"{wf['sharpe']:.2f}" if wf["sharpe"] is not None else "N/A"
    dd_str = f"{wf['max_drawdown']:.2%}" if wf["max_drawdown"] is not None else "N/A"
    wr_str = f"{wf['win_rate']:.1%}" if wf["win_rate"] is not None else "N/A"
    pf_str = f"{wf['profit_factor']:.2f}" if wf["profit_factor"] is not None else "N/A"
    tr_str = f"{wf['total_return']:.2%}" if wf["total_return"] is not None else "N/A"

    print(f"  Sharpe ratio   : {sharpe_str}", file=sys.stderr)
    print(f"  Max drawdown   : {dd_str}", file=sys.stderr)
    print(f"  Total return   : {tr_str}", file=sys.stderr)
    print(f"  Trades         : {wf['n_trades']}", file=sys.stderr)
    print(f"  Win rate       : {wr_str}", file=sys.stderr)
    print(f"  Profit factor  : {pf_str}", file=sys.stderr)

    header("DISCLAIMER")
    print(f"  {sr['disclaimer']}", file=sys.stderr)
    print(sep, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect market regimes (Bull/Bear/Sideways) using Markov analysis.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with price data (accepts relative or absolute paths).",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="yfinance ticker symbol (e.g. ES=F, SPY, BTC-USD).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output JSON to stdout (one JSON object, nothing else on stdout).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="threshold",
        choices=["threshold", "messina", "hmm"],
        help="Analysis engine: threshold (fast, close-only), messina (HMM+12 features), hmm (HMM+44 features). Default: threshold.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size for regime classification (default: 20).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Return threshold for bull/bear classification (default: 0.05).",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=252,
        help="Minimum bars before walk-forward trading starts (default: 252).",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=3,
        help="Number of HMM states (default: 3).",
    )

    args = parser.parse_args()

    # Validate source arguments
    if args.csv is None and args.ticker is None:
        parser.error("one of the arguments --ticker --csv is required")

    if args.csv is not None and args.ticker is not None:
        parser.error("arguments --ticker and --csv are mutually exclusive")

    try:
        # Load data
        source: str
        ohlcv = None
        if args.csv is not None:
            source = args.csv
            prices = load_from_csv(args.csv)
        else:
            source = args.ticker  # type: ignore[assignment]
            import yfinance
            ohlcv_raw = yfinance.download(args.ticker, period="10y", progress=False)
            # Flatten MultiIndex columns from yfinance (Ticker, Price)
            if isinstance(ohlcv_raw.columns, pd.MultiIndex):
                ohlcv_raw.columns = [c[0].lower() for c in ohlcv_raw.columns]
            else:
                ohlcv_raw.columns = [c.lower() for c in ohlcv_raw.columns]
            prices = ohlcv_raw["close"]
            if args.engine in ("messina", "hmm"):
                ohlcv = ohlcv_raw[["open", "high", "low", "close", "volume"]]

        # Guard: need at least 2 price points to compute returns
        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 rows of price data, got {len(prices)}. "
                "Cannot compute returns from a single price point."
            )

        # Build output
        output = pipeline_run(
            prices=prices,
            source=source,
            engine=args.engine,
            window=args.window,
            threshold=args.threshold,
            min_train=args.min_train,
            ohlcv=ohlcv,
            n_states=args.n_states,
        )

        if args.json:
            json.dump(output, sys.stdout, indent=2, allow_nan=False)
            sys.stdout.write("\n")
        else:
            _print_terminal(output)

    except Exception as exc:
        if args.json:
            json.dump({"error": str(exc)}, sys.stdout)
            sys.stdout.write("\n")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
