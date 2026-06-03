#!/usr/bin/env python3
"""HMMMessina parameter sweep across all tickers in test_data/eval-results/.

Phase 1: Run with defaults
Phase 2: Grid search over all tunable parameters
Phase 3: Compare and judge

Outputs JSON results to test_data/eval-results/messina_sweep/ for analysis.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run
from hmm_futures_analysis.utils.logging_config import suppress_stdout_logging

suppress_stdout_logging()

CSV_DIR = Path("test_data/eval-results")
OUT_DIR = Path("test_data/eval-results/messina_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameter grid for Phase 2 ──────────────────────────────────────────
# HMMMMessinaConfig parameters:
#   n_states: int|str = 3, pca_variance: float|None = None
# Pipeline parameters (passed separately): dwell_bars, hysteresis_delta
PHASE2_GRID = {
    "n_states": [2, 3, 4, 5, "auto"],
    "pca_variance": [None, 0.95, 0.99],
    "dwell_bars": [0, "auto", 2, 5],
    "hysteresis_delta": [0.0, 0.1, 0.05, 0.2],
}

EVAL_METRICS = [
    "sharpe", "max_drawdown", "n_trades", "win_rate",
    "profit_factor", "total_return", "signal", "regime",
]


def extract_result(source_name: str, engine_name: str, output) -> dict:
    """Extract evaluation-relevant fields from a PipelineResult."""
    wf = output.walk_forward
    return {
        "ticker": source_name,
        "engine": engine_name,
        "regime": output.current_regime["name"],
        "signal": round(float(output.signal), 6),
        "sharpe": wf["sharpe"],
        "max_drawdown": wf["max_drawdown"],
        "n_trades": wf["n_trades"],
        "win_rate": wf["win_rate"],
        "profit_factor": wf["profit_factor"],
        "total_return": wf["total_return"],
        "n_states": getattr(output.engine_info, "n_states", None)
        if hasattr(output, "engine_info") and hasattr(output.engine_info, "n_states")
        else output.engine_info.get("n_states"),
        "features": getattr(output.engine_info, "features", None)
        if hasattr(output, "engine_info")
        else output.engine_info.get("features"),
        "degenerate": output.engine_info.get("degenerate", False),
        "degeneracy_detail": output.engine_info.get("degeneracy_detail", None),
    }


def fmt_value(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def param_label(config: HMMMMessinaConfig, dwell: int | str, hyst: float | str) -> str:
    parts = [
        f"n_states={config.n_states}",
        f"pca={config.pca_variance}" if config.pca_variance else "pca=None",
        f"dwell={dwell}",
        f"hyst={hyst}",
    ]
    return "|".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: Default run
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72, file=sys.stderr)
print("PHASE 1: HMMMessina with DEFAULT parameters on all tickers", file=sys.stderr)
print("=" * 72, file=sys.stderr)

csv_files = sorted(CSV_DIR.glob("*.csv"))
phase1_results = []
for csv_file in csv_files:
    ticker = csv_file.stem
    print(f"\n  Loading {csv_file.name}…", file=sys.stderr)
    prices, ohlcv, source = load_prices(csv=str(csv_file))

    config = HMMMMessinaConfig()  # All defaults
    t0 = time.monotonic()
    output = pipeline_run(
        prices=prices,
        source=ticker,
        engine_config=config,
        min_train=252,
        ohlcv=ohlcv,
        dwell_bars=0,
        hysteresis_delta=0.1,  # HMMMMessinaConfig default
    )
    elapsed = time.monotonic() - t0

    rec = extract_result(ticker, "messina", output)
    rec["wall_seconds"] = round(elapsed, 3)
    rec["config_key"] = "DEFAULT"
    phase1_results.append(rec)

    print(f"    regime={rec['regime']}  sharpe={rec['sharpe']}  "
          f"ret={rec['total_return']}  trades={rec['n_trades']}  "
          f"win={rec['win_rate']}  pf={rec['profit_factor']}  "
          f"dd={rec['max_drawdown']}  [{elapsed:.1f}s]", file=sys.stderr)

# Save Phase 1 results
with open(OUT_DIR / "phase1_defaults.json", "w") as f:
    json.dump(phase1_results, f, indent=2, allow_nan=False)
print(f"\n✓ Phase 1 saved: {len(phase1_results)} results to {OUT_DIR / 'phase1_defaults.json'}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: Grid search over all parameter combinations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72, file=sys.stderr)
print("PHASE 2: Grid search over HMMMessina parameters", file=sys.stderr)
print("=" * 72, file=sys.stderr)

total_combos = (
    len(PHASE2_GRID["n_states"])
    * len(PHASE2_GRID["pca_variance"])
    * len(PHASE2_GRID["dwell_bars"])
    * len(PHASE2_GRID["hysteresis_delta"])
)
total_runs = total_combos * len(csv_files)
print(f"  Parameter combos: {total_combos}", file=sys.stderr)
print(f"  Tickers: {len(csv_files)} ({', '.join(f.stem for f in csv_files)})", file=sys.stderr)
print(f"  Total runs: {total_runs}", file=sys.stderr)
print(file=sys.stderr)

phase2_results = []
run_count = 0
error_count = 0

for csv_file in csv_files:
    ticker = csv_file.stem
    print(f"\n── {ticker} ──", file=sys.stderr)
    prices, ohlcv, source = load_prices(csv=str(csv_file))

    for n_states, pca_var, dwell, hyst in itertools.product(
        PHASE2_GRID["n_states"],
        PHASE2_GRID["pca_variance"],
        PHASE2_GRID["dwell_bars"],
        PHASE2_GRID["hysteresis_delta"],
    ):
        # Build config with these parameters
        config = HMMMMessinaConfig(n_states=n_states, pca_variance=pca_var)
        label = param_label(config, dwell, hyst)
        run_count += 1

        try:
            t0 = time.monotonic()
            output = pipeline_run(
                prices=prices,
                source=ticker,
                engine_config=config,
                min_train=252,
                ohlcv=ohlcv,
                dwell_bars=dwell,
                hysteresis_delta=hyst,
            )
            elapsed = time.monotonic() - t0

            rec = extract_result(ticker, "messina", output)
            rec["wall_seconds"] = round(elapsed, 3)
            rec["config_key"] = label
            rec["dwell_bars"] = dwell
            rec["hysteresis_delta"] = hyst
            phase2_results.append(rec)

            # Compact progress line
            print(f"  [{run_count:4d}/{total_runs}] {label:40s} → "
                  f"sharpe={rec['sharpe']}  ret={rec['total_return']}  "
                  f"trades={rec['n_trades']}  [{elapsed:.1f}s]", file=sys.stderr)

        except Exception as e:
            error_count += 1
            print(f"  [{run_count:4d}/{total_runs}] {label:40s} → ERROR: {e}",
                  file=sys.stderr)
            phase2_results.append({
                "ticker": ticker,
                "engine": "messina",
                "config_key": label,
                "n_states": n_states,
                "pca_variance": pca_var,
                "dwell_bars": dwell,
                "hysteresis_delta": hyst,
                "error": str(e),
                "sharpe": None,
                "max_drawdown": None,
                "n_trades": None,
                "win_rate": None,
                "profit_factor": None,
                "total_return": None,
                "signal": None,
                "regime": None,
                "wall_seconds": None,
            })

# Save Phase 2 results
with open(OUT_DIR / "phase2_grid.json", "w") as f:
    json.dump(phase2_results, f, indent=2, allow_nan=False)
print(f"\n✓ Phase 2 saved: {len(phase2_results)} results to {OUT_DIR / 'phase2_grid.json'}", file=sys.stderr)
if error_count:
    print(f"  Errors: {error_count}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: Analysis & comparison
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72, file=sys.stderr)
print("PHASE 3: Analysis — Default vs Best vs Aggregate", file=sys.stderr)
print("=" * 72, file=sys.stderr)

# Build per-ticker comparison
analysis = {}
for csv_file in csv_files:
    ticker = csv_file.stem

    # Phase 1 default result
    defaults = [r for r in phase1_results if r["ticker"] == ticker]
    default_rec = defaults[0] if defaults else None

    # Phase 2 results (exclude errors)
    grid_recs = [r for r in phase2_results if r["ticker"] == ticker and r.get("sharpe") is not None]

    if not grid_recs:
        continue

    # Best by Sharpe
    best_sharpe = max(grid_recs, key=lambda r: r["sharpe"] if r["sharpe"] is not None else -999)
    # Best by total return
    best_return = max(grid_recs, key=lambda r: r["total_return"] if r["total_return"] is not None else -999)
    # Best by win rate
    best_winrate = max(grid_recs, key=lambda r: r["win_rate"] if r["win_rate"] is not None else -999)
    # Best by profit factor
    best_pf = max(grid_recs, key=lambda r: r["profit_factor"] if r["profit_factor"] is not None else -999)
    # Lowest drawdown
    best_dd = min(grid_recs, key=lambda r: r["max_drawdown"] if r["max_drawdown"] is not None else 999)

    # Best combo by combined score (z-score aggregation across metrics)
    shaprs = [r["sharpe"] for r in grid_recs if r["sharpe"] is not None]
    returns = [r["total_return"] for r in grid_recs if r["total_return"] is not None]
    dds = [r["max_drawdown"] for r in grid_recs if r["max_drawdown"] is not None]

    def zscore(val, series):
        if val is None or len(series) < 2:
            return 0.0
        mu = sum(series) / len(series)
        sd = (sum((x - mu) ** 2 for x in series) / (len(series) - 1)) ** 0.5
        return (val - mu) / sd if sd > 0 else 0.0

    def composite(r):
        s = 0
        s += zscore(r["sharpe"], shaprs) * 0.30
        s += zscore(r["total_return"], returns) * 0.25
        s += -zscore(r["max_drawdown"], dds) * 0.20  # negate drawdown
        if r["win_rate"] is not None:
            wrs = [x["win_rate"] for x in grid_recs if x["win_rate"] is not None]
            s += zscore(r["win_rate"], wrs) * 0.15
        if r["profit_factor"] is not None:
            pfs = [x["profit_factor"] for x in grid_recs if x["profit_factor"] is not None]
            s += zscore(r["profit_factor"], pfs) * 0.10
        return s

    best_composite = max(grid_recs, key=composite)

    analysis[ticker] = {
        "default": {
            "config": default_rec["config_key"] if default_rec else "N/A",
            "sharpe": default_rec["sharpe"] if default_rec else None,
            "total_return": default_rec["total_return"] if default_rec else None,
            "n_trades": default_rec["n_trades"] if default_rec else None,
            "win_rate": default_rec["win_rate"] if default_rec else None,
            "profit_factor": default_rec["profit_factor"] if default_rec else None,
            "max_drawdown": default_rec["max_drawdown"] if default_rec else None,
        },
        "best_sharpe": {
            "config": best_sharpe["config_key"],
            "sharpe": best_sharpe["sharpe"],
            "total_return": best_sharpe["total_return"],
            "n_trades": best_sharpe["n_trades"],
            "win_rate": best_sharpe["win_rate"],
            "profit_factor": best_sharpe["profit_factor"],
            "max_drawdown": best_sharpe["max_drawdown"],
        },
        "best_return": {
            "config": best_return["config_key"],
            "sharpe": best_return["sharpe"],
            "total_return": best_return["total_return"],
            "n_trades": best_return["n_trades"],
            "win_rate": best_return["win_rate"],
            "profit_factor": best_return["profit_factor"],
            "max_drawdown": best_return["max_drawdown"],
        },
        "best_winrate": {
            "config": best_winrate["config_key"],
            "sharpe": best_winrate["sharpe"],
            "total_return": best_winrate["total_return"],
            "n_trades": best_winrate["n_trades"],
            "win_rate": best_winrate["win_rate"],
            "profit_factor": best_winrate["profit_factor"],
            "max_drawdown": best_winrate["max_drawdown"],
        },
        "best_drawdown": {
            "config": best_dd["config_key"],
            "sharpe": best_dd["sharpe"],
            "total_return": best_dd["total_return"],
            "n_trades": best_dd["n_trades"],
            "win_rate": best_dd["win_rate"],
            "profit_factor": best_dd["profit_factor"],
            "max_drawdown": best_dd["max_drawdown"],
        },
        "best_composite": {
            "config": best_composite["config_key"],
            "sharpe": best_composite["sharpe"],
            "total_return": best_composite["total_return"],
            "n_trades": best_composite["n_trades"],
            "win_rate": best_composite["win_rate"],
            "profit_factor": best_composite["profit_factor"],
            "max_drawdown": best_composite["max_drawdown"],
        },
        "n_runs": len(grid_recs),
        "n_errors": len([r for r in phase2_results if r["ticker"] == ticker and r.get("error")]),
    }

    # Print analysis
    d = analysis[ticker]
    print(f"\n── {ticker} ({d['n_runs']} runs, {d['n_errors']} errors) ──", file=sys.stderr)

    header = f"{'Metric':<18s} {'Default':>12s}  {'Best Sharpe':>12s}  {'Best Return':>12s}  {'Best WinR':>12s}  {'Best DD':>12s}  {'Best Comp':>12s}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"

    for metric in ["sharpe", "total_return", "n_trades", "win_rate", "profit_factor", "max_drawdown"]:
        dv = d["default"].get(metric)
        bsv = d["best_sharpe"].get(metric)
        brv = d["best_return"].get(metric)
        bwv = d["best_winrate"].get(metric)
        bddv = d["best_drawdown"].get(metric)
        bcv = d["best_composite"].get(metric)
        print(f"  {metric:<16s} {fmt(dv):>12s}  {fmt(bsv):>12s}  {fmt(brv):>12s}  {fmt(bwv):>12s}  {fmt(bddv):>12s}  {fmt(bcv):>12s}", file=sys.stderr)

    print(f"\n  Config keys:", file=sys.stderr)
    print(f"    Best Sharpe:    {d['best_sharpe']['config']}", file=sys.stderr)
    print(f"    Best Return:    {d['best_return']['config']}", file=sys.stderr)
    print(f"    Best Composite: {d['best_composite']['config']}", file=sys.stderr)

# Save analysis
with open(OUT_DIR / "phase3_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, allow_nan=False)
print(f"\n✓ Phase 3 analysis saved to {OUT_DIR / 'phase3_analysis.json'}", file=sys.stderr)

# ── Overall summary across all tickers ──
print("\n" + "=" * 72, file=sys.stderr)
print("OVERALL SUMMARY", file=sys.stderr)
print("=" * 72, file=sys.stderr)

all_defaults = phase1_results
all_best_composite = [analysis[t]["best_composite"] for t in analysis]

# Aggregate default metrics (mean across tickers)
def safe_mean(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def safe_sum(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    if not vals:
        return None
    return sum(vals)

print(f"\n  {'Metric':<20s} {'Default (avg)':>14s}  {'Best Comp (avg)':>16s}  {'Improvement':>12s}", file=sys.stderr)
print(f"  {'-'*20}  {'-'*14}  {'-'*16}  {'-'*12}", file=sys.stderr)

for metric in ["sharpe", "total_return", "win_rate", "profit_factor", "max_drawdown", "n_trades"]:
    dm = safe_mean(all_defaults, metric)
    bm = safe_mean(all_best_composite, metric)
    if dm is not None and bm is not None and dm != 0:
        imp = (bm - dm) / abs(dm) * 100
        imp_str = f"{imp:+.1f}%"
    elif dm is not None and bm is not None:
        imp_str = f"{bm-dm:+.4f}"
    else:
        imp_str = "N/A"
    print(f"  {metric:<20s} {fmt(dm):>14s}  {fmt(bm):>16s}  {imp_str:>12s}", file=sys.stderr)

# Which parameters were most commonly optimal?
print(f"\n  Parameter frequency among 'best composite' configs:", file=sys.stderr)
param_freq = {"n_states": {}, "pca_variance": {}, "dwell_bars": {}, "hysteresis_delta": {}}
for ticker, info in analysis.items():
    cfg = info["best_composite"]["config"]
    parts = cfg.split("|")
    for p in parts:
        k, v = p.split("=", 1)
        if k in param_freq:
            param_freq[k][v] = param_freq[k].get(v, 0) + 1

for param, freqs in param_freq.items():
    total = sum(freqs.values())
    sorted_freqs = sorted(freqs.items(), key=lambda x: -x[1])
    freq_str = ", ".join(f"{v}={c}" for v, c in sorted_freqs)
    print(f"    {param:<16s}: {freq_str}", file=sys.stderr)

print(f"\n✓ Complete. Results in {OUT_DIR}/", file=sys.stderr)
