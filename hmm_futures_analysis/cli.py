#!/usr/bin/env python3
"""Regime detection entry point for the hmm-regime-detection skill.

CLI for detecting market regimes (Bull/Bear/Sideways) using one of five
independent engines: threshold (fast, close-only), messina (HMM + 19
Messina features), hmm (HMM + ~50 generic features), robust_hmm (HMM +
robust outlier-resistant emissions), or fshmm (HMM + feature saliency).

Usage:
    hmm-regime --csv BTC.csv --json
    hmm-regime --ticker ES=F --json --engine hmm
    hmm-regime --csv data.csv --engine threshold
    hmm-regime --csv data.csv --engine robust_hmm --robust-method mcd
    hmm-regime --csv data.csv --engine fshmm --saliency-threshold 0.3
    ./scripts/run.sh --csv BTC.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .data_processing.csv_auto_detect import load_prices
from .eval import (
    ALL_ENGINES as EVAL_ALL_ENGINES,
    run_eval_csv,
    run_eval_tickers,
)
from .presenter import (
    format_eval,
    format_pipeline,
    limit_transitions,
    serialize_pipeline,
)
from .regime.engine_config_builder import build_engine_config
from .regime.pipeline import run as pipeline_run
from .utils.logging_config import suppress_stdout_logging

_FRAMEWORK_VERSION = "hmm_test v0.2.0"


def _write_saliency_csv(output, path: str) -> None:
    """Write fshmm saliency weights to a CSV file."""
    import csv as csv_mod

    saliency = output.engine_info.get("feature_saliency")
    selected = output.engine_info.get("selected_features")
    if saliency is None:
        return

    with open(path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["feature_index", "saliency_weight", "selected"])
        for i, w in enumerate(saliency):
            is_sel = "yes" if selected and f"f{i}" in selected else "no"
            writer.writerow([i, f"{w:.6f}", is_sel])


def _parse_n_states(value: str) -> str | int:
    """Parse --n-states: accept 'auto' or an integer >= 2."""
    if value == "auto":
        return "auto"
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--n-states must be 'auto' or an integer, got {value!r}"
        )
    if iv < 2:
        raise argparse.ArgumentTypeError(f"--n-states must be >= 2, got {iv}")
    return iv


def _parse_dwell_bars(value: str) -> str | int:
    """Parse --dwell-bars: accept 'auto' or a non-negative integer."""
    if value == "auto":
        return "auto"
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--dwell-bars must be 'auto' or an integer, got {value!r}"
        )
    if iv < 0:
        raise argparse.ArgumentTypeError(f"--dwell-bars must be >= 0, got {iv}")
    return iv


def _parse_hysteresis(value: str) -> str | float:
    """Parse --hysteresis: accept 'auto' or a float in [0, 1)."""
    if value == "auto":
        return "auto"
    try:
        fv = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--hysteresis must be 'auto' or a number, got {value!r}"
        )
    if fv < 0.0 or fv >= 1.0:
        raise argparse.ArgumentTypeError(f"--hysteresis must be in [0, 1), got {fv}")
    return fv


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
        choices=["threshold", "messina", "hmm", "robust_hmm", "fshmm"],
        help="Analysis engine: threshold (fast, close-only), messina (HMM+19 features), hmm (HMM+50 features), robust_hmm (HMM with outlier-resistant emissions), fshmm (HMM+feature saliency). Default: threshold.",
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
        type=_parse_n_states,
        default=3,
        help="Number of HMM states: 'auto' for BIC selection, or an integer >= 2 (default: 3).",
    )
    parser.add_argument(
        "--dwell-bars",
        type=_parse_dwell_bars,
        default="auto",
        help="Dwell-time filter: require N consecutive same-regime bars before switching position. Accepts 'auto' for engine defaults (default: auto).",
    )
    parser.add_argument(
        "--hysteresis",
        type=_parse_hysteresis,
        default="auto",
        help="Hysteresis filter: require posterior probability margin > D to switch regime. Accepts 'auto' for engine defaults (default: auto).",
    )
    parser.add_argument(
        "--robust-method",
        type=str,
        default="huber",
        choices=["huber", "mcd"],
        help="Robust estimation method for robust_hmm engine: huber (Huber IRLS) or mcd (MinCovDet). Default: huber.",
    )

    parser.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help="PCA whitening: retain this fraction of variance (e.g. 0.95). When omitted, uses the engine's dataclass default (messina=0.95, robust_hmm=0.90, hmm=None, fshmm=None).",
    )
    parser.add_argument(
        "--saliency-threshold",
        type=float,
        default=0.5,
        help="Feature saliency pruning threshold for fshmm engine (default: 0.5).",
    )
    parser.add_argument(
        "--saliency-output",
        type=str,
        default=None,
        help="Save fshmm saliency weights to CSV file.",
    )

    # --- Eval mode arguments ---
    eval_group = parser.add_argument_group(
        "eval mode",
        "Run multiple engines across multiple tickers/CSVs.",
    )
    eval_group.add_argument(
        "--eval-tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list for batch eval (e.g. CRM,0700.HK,SPY).",
    )
    eval_group.add_argument(
        "--eval-csv",
        type=str,
        default=None,
        help="Directory of CSV files for batch eval (filename stem = ticker).",
    )
    eval_group.add_argument(
        "--eval-engines",
        type=str,
        default=None,
        help=(
            "Comma-separated engine list for eval mode "
            f"(default: all). Choices: {','.join(EVAL_ALL_ENGINES)}."
        ),
    )
    eval_group.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached yfinance ticker CSVs (default: ~/.cache/hmm-regime/tickers/).",
    )
    eval_group.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Force re-download and overwrite cached ticker data.",
    )
    eval_group.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass the ticker cache entirely — do not read or write.",
    )
    eval_group.add_argument(
        "--eval-cache-dir",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--duration-forecast",
        action="store_true",
        default=False,
        help="Enable regime duration forecasting via Weibull survival analysis (default: disabled).",
    )
    parser.add_argument(
        "--duration-model",
        type=str,
        default="weibull",
        choices=["weibull", "cox"],
        help="Survival model for duration forecasting: weibull (default) or cox (requires lifelines).",
    )
    parser.add_argument(
        "--transitions",
        type=int,
        default=None,
        help="Show N most recent regime transitions in terminal output (default: disabled). 0 shows all.",
    )
    parser.add_argument(
        "--reverse-classify",
        action="store_true",
        default=False,
        help=(
            "Reverse walk-forward classify direction so degeneration hits old bars "
            "instead of recent ones. Lookahead warning issued (Issue #102)."
        ),
    )

    args = parser.parse_args()

    # --- Determine mode: eval vs single-run ---
    eval_mode = args.eval_tickers is not None or args.eval_csv is not None
    single_mode = args.csv is not None or args.ticker is not None

    if eval_mode and single_mode:
        parser.error(
            "eval flags (--eval-tickers, --eval-csv) are mutually exclusive "
            "with single-run flags (--ticker, --csv)"
        )

    if not eval_mode and not single_mode:
        parser.error("provide one of: --ticker, --csv, --eval-tickers, or --eval-csv")

    # --- Eval mode ---
    if eval_mode:
        if args.eval_tickers is not None and args.eval_csv is not None:
            parser.error("--eval-tickers and --eval-csv are mutually exclusive")

        # Parse engine filter
        engines = EVAL_ALL_ENGINES
        if args.eval_engines is not None:
            requested = [e.strip() for e in args.eval_engines.split(",")]
            invalid = set(requested) - set(EVAL_ALL_ENGINES)
            if invalid:
                parser.error(
                    f"unknown engines: {','.join(sorted(invalid))}. "
                    f"Valid: {','.join(EVAL_ALL_ENGINES)}"
                )
            engines = tuple(requested)

        # Backwards compatibility: --eval-cache-dir maps to --cache-dir
        cache_dir = args.cache_dir if args.cache_dir else args.eval_cache_dir

        try:
            if args.json:
                suppress_stdout_logging()

            if args.eval_csv:
                results = run_eval_csv(
                    csv_dir=args.eval_csv,
                    engines=engines,
                    min_train=args.min_train,
                )
            else:
                tickers = tuple(t.strip() for t in args.eval_tickers.split(","))
                results = run_eval_tickers(
                    tickers=tickers,
                    cache_dir=cache_dir,
                    refresh=args.refresh,
                    no_cache=args.no_cache,
                    engines=engines,
                    min_train=args.min_train,
                )

            if args.json:
                print(format_eval(results, fmt="json"))
            else:
                print(format_eval(results, fmt="table"), file=sys.stderr)

        except Exception as exc:
            if args.json:
                json.dump({"error": str(exc)}, sys.stdout)
                sys.stdout.write("\n")
            else:
                print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-run mode (original behaviour) ---
    try:
        # In JSON mode, suppress logging to stdout so only the JSON object is emitted
        if args.json:
            suppress_stdout_logging()

        # Load data
        prices, ohlcv, source = load_prices(
            csv=args.csv,
            ticker=args.ticker,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            no_cache=args.no_cache,
        )

        # Build engine config from CLI args
        engine_config = build_engine_config(args)

        # Build output
        output = pipeline_run(
            prices=prices,
            source=source,
            engine_config=engine_config,
            min_train=args.min_train,
            ohlcv=ohlcv,
            dwell_bars=args.dwell_bars,
            hysteresis_delta=args.hysteresis,
            duration_forecast=args.duration_forecast,
            duration_model=args.duration_model,
        )

        if args.json:
            json.dump(serialize_pipeline(output, transitions_limit=args.transitions), sys.stdout, indent=2, allow_nan=False)
            sys.stdout.write("\n")
        else:
            print(format_pipeline(output, transitions_limit=args.transitions), file=sys.stderr, end="")

        # Write saliency weights CSV if requested
        if args.saliency_output and args.engine == "fshmm":
            _write_saliency_csv(output, args.saliency_output)

    except Exception as exc:
        if args.json:
            json.dump({"error": str(exc)}, sys.stdout)
            sys.stdout.write("\n")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
