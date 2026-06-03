#!/usr/bin/env python3
"""Surgical parameter sensitivity sweep for HMMMessina.

One-at-a-time variations from default to identify which parameters matter,
keeping compute to O(N_params) instead of O(N_params^4) = 12 vs 192 runs/ticker.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run
from hmm_futures_analysis.utils.logging_config import suppress_stdout_logging

suppress_stdout_logging()

CSV_DIR = Path("test_data/eval-results")
OUT_DIR = Path("test_data/eval-results/messina_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default: n_states=3, pca=None, dwell=0, hyst=0.1
# Sensitivities: vary ONE param at a time
SENSITIVITY_RUNS = {
    "n_states=4": HMMMMessinaConfig(n_states=4, pca_variance=None),
    "n_states=5": HMMMMessinaConfig(n_states=5, pca_variance=None),
    "n_states=auto": HMMMMessinaConfig(n_states="auto", pca_variance=None),
    "pca=0.95": HMMMMessinaConfig(n_states=3, pca_variance=0.95),
    "pca=0.99": HMMMMessinaConfig(n_states=3, pca_variance=0.99),
}

# Pipeline-level params varied separately
PIPELINE_VARIANTS = [
    ("dwell=auto|hyst=0.0",  "auto", 0.0),
    ("dwell=2|hyst=0.0",     2,    0.0),
    ("dwell=5|hyst=0.0",     5,    0.0),
    ("dwell=0|hyst=0.05",    0,    0.05),
]

all_results = []
csv_files = sorted(CSV_DIR.glob("*.csv"))

for csv_file in csv_files:
    ticker = csv_file.stem
    print(f"\n{'='*60}\n{'-'*60}", flush=True)
    print(f"TICKER: {ticker}", flush=True)
    print(f"{'-'*60}", flush=True)
    prices, ohlcv, source = load_prices(csv=str(csv_file))

    # ── Phase 2a: Sensitivity sweep ──
    sensitivity_runs = []

    default_config = HMMMMessinaConfig()
    sensitivity_runs.append(("DEFAULT", default_config, 0, 0.1))
    sensitivity_runs.append(("DEFAULT+hyst0", default_config, 0, 0.0))

    # Config-only variations (n_states, pca) — hyst=0.0 to actually see trades
    for label, cfg in SENSITIVITY_RUNS.items():
        sensitivity_runs.append((f"{label}+hyst0", cfg, 0, 0.0))  # hyst=0.0

    # Pipeline-level variations (dwell, hyst) — keep default config
    for label, dwell, hyst in PIPELINE_VARIANTS:
        sensitivity_runs.append((label, HMMMMessinaConfig(), dwell, hyst))

    # ── Also try best combo: n_states=auto, pca=0.95, dwell=0, hyst=0.0
    sensitivity_runs.append(("best_guess", HMMMMessinaConfig(n_states="auto", pca_variance=0.95), 0, 0.0))

    total = len(sensitivity_runs)
    print(f"  Runs: {total}\n", flush=True)

    for i, (label, config, dwell, hyst) in enumerate(sensitivity_runs, 1):
        t0 = time.monotonic()
        try:
            output = pipeline_run(
                prices=prices,
                source=ticker,
                engine_config=config,
                min_train=252,
                ohlcv=ohlcv,
                dwell_bars=dwell,
                hysteresis_delta=hyst,
                profile=False,
            )
            elapsed = time.monotonic() - t0
            wf = output.walk_forward
            rec = {
                "ticker": ticker,
                "engine": "messina",
                "config": label,
                "n_states": str(config.n_states),
                "pca_variance": config.pca_variance,
                "dwell_bars": dwell,
                "hysteresis_delta": hyst,
                "regime": output.current_regime["name"],
                "signal": round(float(output.signal), 6),
                "sharpe": wf["sharpe"],
                "max_drawdown": wf["max_drawdown"],
                "n_trades": wf["n_trades"],
                "win_rate": wf["win_rate"],
                "profit_factor": wf["profit_factor"],
                "total_return": wf["total_return"],
                "n_states_resolved": output.engine_info.get("n_states"),
                "degenerate": output.engine_info.get("degenerate", False),
                "wall_seconds": round(elapsed, 3),
                "error": None,
            }
        except Exception as e:
            elapsed = time.monotonic() - t0
            rec = {
                "ticker": ticker,
                "engine": "messina",
                "config": label,
                "error": str(e),
                "wall_seconds": round(elapsed, 3),
            }

        all_results.append(rec)

        sharpe_s = f"{rec.get('sharpe', 'ERR'):.4f}" if rec.get('sharpe') is not None else "NONE"
        ret_s = f"{rec.get('total_return', 'ERR'):.4f}" if rec.get('total_return') is not None else "NONE"
        trd_s = f"{rec.get('n_trades', 'ERR')}" if rec.get('n_trades') is not None else "NONE"
        err_s = f"  ERROR: {rec.get('error', '')}" if rec.get('error') else ""
        print(f"  [{i:2d}/{total}] {label:30s} → sharpe={sharpe_s} ret={ret_s} trades={trd_s} [{elapsed:.1f}s]{err_s}", flush=True)

# ── Save all results ──
with open(OUT_DIR / "phase2_sensitivity.json", "w") as f:
    json.dump(all_results, f, indent=2, allow_nan=False)

print(f"\n{'='*60}", flush=True)
print(f"Saved {len(all_results)} results to {OUT_DIR / 'phase2_sensitivity.json'}", flush=True)

# ── Analysis ──
print(f"\n{'='*60}", flush=True)
print("QUICK ANALYSIS", flush=True)
print(f"{'='*60}", flush=True)

for csv_file in csv_files:
    ticker = csv_file.stem
    tk_results = [r for r in all_results if r["ticker"] == ticker and r.get("error") is None]
    if not tk_results:
        continue

    print(f"\n── {ticker} ({len(tk_results)} runs) ──", flush=True)

    # Default (hyst=0.1, no hyst)
    default_hyst = [r for r in tk_results if r["config"] == "DEFAULT"]
    default_nohyst = [r for r in tk_results if r["config"] == "DEFAULT+hyst0"]

    # Best by each metric
    def best_by(key, rev=True):
        valid = [r for r in tk_results if r.get(key) is not None]
        return max(valid, key=lambda r: r[key]) if valid else None

    best_sharpe = best_by("sharpe")
    best_ret = best_by("total_return")
    best_wr = best_by("win_rate")
    best_pf = best_by("profit_factor")
    lowest_dd = min(tk_results, key=lambda r: r.get("max_drawdown") if r.get("max_drawdown") is not None else 999)

    fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"

    print(f"  {'Metric':<16s} {'Default':>10s} {'Def+hyst0':>10s} {'Best Shar':>10s} {'Best Ret':>10s}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for m in ["sharpe", "total_return", "n_trades", "win_rate", "profit_factor", "max_drawdown"]:
        dv = default_hyst[0].get(m) if default_hyst else None
        dn = default_nohyst[0].get(m) if default_nohyst else None
        bs = best_sharpe.get(m) if best_sharpe else None
        br = best_ret.get(m) if best_ret else None
        print(f"  {m:<16s} {fmt(dv):>10s}  {fmt(dn):>10s}  {fmt(bs):>10s}  {fmt(br):>10s}", flush=True)

    if best_sharpe:
        print(f"\n  Best config (Sharpe): {best_sharpe['config']}", flush=True)
        print(f"    → n_states={best_sharpe['n_states']} pca={best_sharpe['pca_variance']} dwell={best_sharpe['dwell_bars']} hyst={best_sharpe['hysteresis_delta']}", flush=True)
    if best_ret:
        print(f"  Best config (Return): {best_ret['config']}", flush=True)

print(f"\nDone. All results in {OUT_DIR}/", flush=True)
