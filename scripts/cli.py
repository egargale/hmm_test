#!/usr/bin/env python3
"""Regime detection entry point for the hmm-regime-detection skill.

CLI for detecting market regimes (Bull/Bear/Sideways) using threshold-based
classification with optional Hidden Markov Model (HMM) analysis.

Usage:
    python scripts/regime.py --csv BTC.csv --json
    python scripts/regime.py --ticker ES=F --json --window 10 --threshold 0.03
    python scripts/regime.py --csv data.csv --no-hmm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure scripts/ directory is on sys.path so that sibling packages resolve
# when this file is executed directly (e.g. `python scripts/regime.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from data_processing.csv_auto_detect import load_from_csv, load_from_yfinance
from regime.hmm_adapter import run_hmm_regime
from regime.markov_chain import (
    build_transition_matrix,
    classify_regimes,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
    forecast_n_steps,
)
from regime.walk_forward import walk_forward_backtest

_STATE_NAMES = ("bear", "sideways", "bull")
_FRAMEWORK_VERSION = "hmm_test v0.1.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)


def _probs_to_dict(probs: np.ndarray) -> dict[str, float]:
    """Convert a 3-element probability array to {bear, sideways, bull} dict."""
    return {
        "bear": float(probs[0]),
        "sideways": float(probs[1]),
        "bull": float(probs[2]),
    }


def _nan_to_none(value: float) -> float | None:
    """Replace NaN with None for JSON serialisation."""
    return None if isinstance(value, float) and math.isnan(value) else value


def _build_output(
    prices,
    source: str,
    window: int,
    threshold: float,
    min_train: int,
    use_hmm: bool,
    n_states: int,
) -> dict:
    """Build the full JSON-compatible output dictionary."""
    returns = prices.pct_change().dropna()

    # --- Threshold-based regime classification ---
    regimes = classify_regimes(returns, window=window, threshold=threshold)
    transmat = build_transition_matrix(regimes)
    stationary = compute_stationary_distribution(transmat)
    persistence = compute_persistence_diagonal(transmat)

    last_regime = int(regimes[-1])
    current_probs = transmat[last_regime]
    signal = compute_signal(current_probs)

    # Regime counts
    unique, counts = np.unique(regimes, return_counts=True)
    regime_counts_map: dict[str, int] = {"bear": 0, "sideways": 0, "bull": 0}
    for s, c in zip(unique, counts):
        regime_counts_map[_STATE_NAMES[s]] = int(c)

    # Date boundaries
    date_start: str = ""
    date_end: str = ""
    try:
        date_start = str(prices.index[0].date())
        date_end = str(prices.index[-1].date())
    except Exception:
        pass

    # Forecasts
    forecast_1 = _probs_to_dict(forecast_n_steps(transmat, current_probs, 1))
    forecast_5 = _probs_to_dict(forecast_n_steps(transmat, current_probs, 5))
    forecast_20 = _probs_to_dict(forecast_n_steps(transmat, current_probs, 20))

    # Walk-forward backtest
    wf = walk_forward_backtest(prices, window=window, threshold=threshold, min_train=min_train)
    walk_forward = {
        "sharpe": _nan_to_none(wf["sharpe"]),
        "max_drawdown": _nan_to_none(wf["max_drawdown"]),
        "n_trades": wf["n_trades"],
    }

    # HMM (optional)
    hmm_result: dict
    if use_hmm:
        try:
            hmm_result = run_hmm_regime(prices, n_states=n_states)
        except Exception as exc:
            hmm_result = {"available": False, "reason": str(exc)}
    else:
        hmm_result = {"available": False, "reason": "HMM disabled via --no-hmm"}

    output: dict = {
        "source": source,
        "rows": len(prices),
        "date_start": date_start,
        "date_end": date_end,
        "params": {
            "window": window,
            "threshold": threshold,
            "method": "threshold",
        },
        "states": [
            {"name": "bear", "index": 0},
            {"name": "sideways", "index": 1},
            {"name": "bull", "index": 2},
        ],
        "current_regime": {
            "name": _STATE_NAMES[last_regime],
            "index": last_regime,
        },
        "next_state_probabilities": _probs_to_dict(current_probs),
        "signal": signal,
        "transition_matrix": transmat.tolist(),
        "persistence_diagonal": persistence,
        "stationary_distribution": _probs_to_dict(stationary),
        "walk_forward": walk_forward,
        "hmm": hmm_result,
        "hmm_test_extras": {
            "n_states": n_states,
            "method": "threshold",
            "data_points": len(prices),
            "regime_counts": regime_counts_map,
        },
        "forecast": {
            "1_step": forecast_1,
            "5_step": forecast_5,
            "20_step": forecast_20,
        },
        "framework": _FRAMEWORK_VERSION,
        "disclaimer": _DISCLAIMER,
    }
    return output


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
    print(f"  Data points : {sr['rows']}", file=sys.stderr)
    print(f"  Date range  : {sr['date_start']} → {sr['date_end']}", file=sys.stderr)
    print(f"  Window      : {sr['params']['window']}", file=sys.stderr)
    print(f"  Threshold   : {sr['params']['threshold']}", file=sys.stderr)
    print(f"  Method      : {sr['params']['method']}", file=sys.stderr)

    header("CURRENT REGIME")
    cr = sr["current_regime"]
    print(f"  Regime      : {cr['name'].upper()} (index {cr['index']})", file=sys.stderr)
    print(f"  Signal      : {sr['signal']:+.4f}", file=sys.stderr)

    header("REGIME DISTRIBUTION")
    rc = sr["hmm_test_extras"]["regime_counts"]
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
    print(f"  Sharpe ratio   : {sharpe_str}", file=sys.stderr)
    print(f"  Max drawdown   : {dd_str}", file=sys.stderr)
    print(f"  Trades         : {wf['n_trades']}", file=sys.stderr)

    header("HMM ANALYSIS")
    h = sr["hmm"]
    print(f"  Available      : {h.get('available', False)}", file=sys.stderr)
    if not h.get("available"):
        print(f"  Reason         : {h.get('reason', 'unknown')}", file=sys.stderr)
    elif "transition_matrix" in h:
        print(f"  States         : {h.get('regimes', [''])[0] if h.get('regimes') else 'N/A'}", file=sys.stderr)
    if "caveat" in h:
        print(f"  Caveat         : {h['caveat']}", file=sys.stderr)

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
        "--hmm",
        action="store_true",
        default=True,
        dest="use_hmm",
        help="Enable HMM mode (default).",
    )
    parser.add_argument(
        "--no-hmm",
        action="store_false",
        dest="use_hmm",
        help="Disable HMM mode (threshold only).",
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
        if args.csv is not None:
            source = args.csv
            prices = load_from_csv(args.csv)
        else:
            source = args.ticker  # type: ignore[assignment]
            prices = load_from_yfinance(args.ticker)

        # Guard: need at least 2 price points to compute returns
        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 rows of price data, got {len(prices)}. "
                "Cannot compute returns from a single price point."
            )

        # Build output
        output = _build_output(
            prices=prices,
            source=source,
            window=args.window,
            threshold=args.threshold,
            min_train=args.min_train,
            use_hmm=args.use_hmm,
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
