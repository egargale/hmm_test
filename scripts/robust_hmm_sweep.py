#!/usr/bin/env python3
"""Evaluate RobustHMMEngine across all tickers.

Phase 1: Default params (n_states=3, pca_variance=None, robust_method="huber")
Phase 2: Parameter sweep — huber only, all n_states × pca combos
Phase 3: Judgment summary

MCD is skipped because it's slow (90-105s/run) and produces degenerate
1-trade results, and huber is the default/canonical robust method.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run
from hmm_futures_analysis.utils.logging_config import suppress_stdout_logging

suppress_stdout_logging()

CSV_DIR = Path("test_data/eval-results")
OUT_DIR = Path("test_data/eval-results/robust_hmm_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all ticker CSVs (only .csv files, skip subdirs)
csv_files = sorted(f for f in CSV_DIR.glob("*.csv") if f.suffix == ".csv")
tickers = sorted({f.stem for f in csv_files})

print(f"Tickers ({len(tickers)}): {', '.join(tickers)}", flush=True)

HYSTERESIS = 0.0
DWELL = 0
MIN_TRAIN = 252


def run_ticker(csv_file: Path, config: RobustHMMConfig, label: str) -> dict | None:
    ticker = csv_file.stem
    t0 = time.monotonic()
    try:
        prices, ohlcv, source = load_prices(csv=str(csv_file))
        output = pipeline_run(
            prices=prices,
            source=ticker,
            engine_config=config,
            min_train=MIN_TRAIN,
            ohlcv=ohlcv,
            dwell_bars=DWELL,
            hysteresis_delta=HYSTERESIS,
            profile=False,
        )
        elapsed = time.monotonic() - t0
        wf = output.walk_forward
        rec = {
            "ticker": ticker,
            "engine": "robust_hmm",
            "config": label,
            "n_states": str(config.n_states),
            "pca_variance": config.pca_variance,
            "robust_method": config.robust_method,
            "dwell_bars": DWELL,
            "hysteresis_delta": HYSTERESIS,
            "n_states_resolved": output.engine_info.get("n_states"),
            "degenerate": output.engine_info.get("degenerate_fit", False),
            "regime": output.current_regime["name"],
            "signal": round(float(output.signal), 6),
            "sharpe": wf["sharpe"],
            "max_drawdown": wf["max_drawdown"],
            "n_trades": wf["n_trades"],
            "win_rate": wf["win_rate"],
            "profit_factor": wf["profit_factor"],
            "total_return": wf["total_return"],
            "wall_seconds": round(elapsed, 3),
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        rec = {
            "ticker": ticker,
            "engine": "robust_hmm",
            "config": label,
            "error": str(e)[:200],
            "wall_seconds": round(elapsed, 3),
        }
    return rec


def print_rec(rec: dict, idx: int, total: int) -> None:
    if rec.get("error"):
        print(f"  [{idx:3d}/{total}] {rec['config']:36s} → ERROR: {rec['error'][:60]} [{rec['wall_seconds']:.1f}s]",
              flush=True)
    else:
        sh = f"{rec['sharpe']:.4f}" if rec['sharpe'] is not None else "NONE"
        rt = f"{rec['total_return']:.4f}" if rec['total_return'] is not None else "NONE"
        wr = f"{rec['win_rate']:.3f}" if rec['win_rate'] is not None else "NONE"
        print(f"  [{idx:3d}/{total}] {rec['config']:36s} → sharpe={sh}  ret={rt}  trades={rec['n_trades']:>2d}  "
              f"win={wr}  regime={rec['regime']:>8s} [{rec['wall_seconds']:.1f}s]", flush=True)


all_results = []
run_count = 0

# Parameter grid (huber only — MCD is slow and degenerate)
# 3 × 4 = 12 combos per ticker, +1 default = 13 per ticker
N_STATES_OPTIONS = [3, 5, "auto"]
PCA_OPTIONS = [None, 0.90, 0.95, 0.99]
ROBUST_METHOD = "huber"

TOTAL_DEFAULT = len(csv_files)
TOTAL_SWEEP = len(csv_files) * len(N_STATES_OPTIONS) * len(PCA_OPTIONS)
TOTAL = TOTAL_DEFAULT + TOTAL_SWEEP

# ═══════════════════════════════════════════════
# Phase 1: DEFAULT config on all tickers
# ═══════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"PHASE 1: Default RobustHMMConfig on all {len(csv_files)} tickers", flush=True)
print(f"{'='*70}", flush=True)

default_config = RobustHMMConfig(n_states=3, pca_variance=None, robust_method="huber")

for i, csv_file in enumerate(csv_files, 1):
    rec = run_ticker(csv_file, default_config, "DEFAULT")
    all_results.append(rec)
    run_count += 1
    print_rec(rec, run_count, TOTAL)

# ═══════════════════════════════════════════════
# Phase 2: Parameter sweep
# ═══════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print(f"PHASE 2: Parameter sweep ({len(N_STATES_OPTIONS)} states × "
      f"{len(PCA_OPTIONS)} pca × {len(csv_files)} tickers = {TOTAL_SWEEP} runs)",
      flush=True)
print(f"{'='*70}", flush=True)

for csv_file in csv_files:
    ticker = csv_file.stem
    print(f"\n── {ticker} ──", flush=True)

    for n_states in N_STATES_OPTIONS:
        for pca_var in PCA_OPTIONS:
            # Skip default config (already in phase 1)
            if n_states == 3 and pca_var is None:
                continue

            config = RobustHMMConfig(
                n_states=n_states,
                pca_variance=pca_var,
                robust_method=ROBUST_METHOD,
            )

            ns = "ns=auto" if n_states == "auto" else f"ns={n_states}"
            pca = "pca=no" if pca_var is None else f"pca={pca_var}"
            label = f"{ns}_{pca}"

            run_count += 1
            rec = run_ticker(csv_file, config, label)
            all_results.append(rec)
            print_rec(rec, run_count, TOTAL)

# ── Save all results ──
results_path = OUT_DIR / "robust_hmm_sweep_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, allow_nan=False)
print(f"\nSaved results to {results_path}", flush=True)

# ── ANALYSIS ──
print(f"\n{'='*100}", flush=True)
print("ANALYSIS: RobustHMMEngine — Default vs Parameter Tuning", flush=True)
print(f"{'='*100}", flush=True)

ok = [r for r in all_results if r.get("error") is None and r.get("sharpe") is not None]

# --- Per-ticker: Best config vs Default ---
print(f"\n{'Ticker':<12s} {'Default Sharpe':>16s} {'Default Ret':>14s} "
      f"{'Best Config':>38s} {'Best Sharpe':>14s} {'Best Ret':>14s} {'Δ Sharpe':>10s}", flush=True)
print(f"{'-'*12} {'-'*16} {'-'*14} {'-'*38} {'-'*14} {'-'*14} {'-'*10}", flush=True)

per_ticker = {}
for csv_file in csv_files:
    ticker = csv_file.stem
    tk = sorted(
        [r for r in ok if r["ticker"] == ticker],
        key=lambda r: r["sharpe"] if r["sharpe"] is not None else -999,
        reverse=True,
    )
    if not tk:
        continue

    dflt = next((r for r in tk if r["config"] == "DEFAULT"), None)
    best = tk[0]
    dsh = dflt["sharpe"] if dflt else None
    drt = dflt["total_return"] if dflt else None
    delta = (best["sharpe"] - dsh) if (best["sharpe"] is not None and dsh is not None) else None
    delta_s = f"{delta:+.4f}" if delta is not None else "N/A"

    if dsh is None:
        print(f"{ticker:<12s} {'ERROR':>16s} {'ERROR':>14s} "
              f"{best['config']:>38s} {best['sharpe']:>14.4f} {best['total_return']:>14.4f} {'N/A':>10s}", flush=True)
    else:
        print(f"{ticker:<12s} {dsh:>16.4f} {drt:>14.4f} "
              f"{best['config']:>38s} {best['sharpe']:>14.4f} {best['total_return']:>14.4f} {delta_s:>10s}", flush=True)

    per_ticker[ticker] = {
        "default_sharpe": dsh, "default_return": drt,
        "best_config": best["config"], "best_sharpe": best["sharpe"],
        "best_params": {"n_states": best["n_states"], "pca_variance": best["pca_variance"]},
        "sharpe_delta": delta,
    }

# --- Parameter value frequency in top-3 ---
param_freq = {}
for csv_file in csv_files:
    ticker = csv_file.stem
    tk = sorted(
        [r for r in ok if r["ticker"] == ticker],
        key=lambda r: r["sharpe"] if r["sharpe"] is not None else -999,
        reverse=True,
    )[:3]
    for r in tk:
        for k in ["n_states", "pca_variance"]:
            v = r.get(k)
            key = f"n_states={v}" if k == "n_states" else (f"pca={v}" if v is not None else "pca=none")
            param_freq[key] = param_freq.get(key, 0) + 1

print(f"\n  Parameter value frequency in TOP-3 configs per ticker:", flush=True)
for pv, cnt in sorted(param_freq.items(), key=lambda x: -x[1]):
    print(f"    {pv:<20s}  {cnt:2d} / {(len(csv_files)*3)}", flush=True)

# --- Config ranking by avg Sharpe ---
config_perf = {}
for r in ok:
    c = r["config"]
    if c not in config_perf:
        config_perf[c] = {"sh": [], "ret": []}
    if r["sharpe"] is not None:
        config_perf[c]["sh"].append(r["sharpe"])
    if r["total_return"] is not None:
        config_perf[c]["ret"].append(r["total_return"])

ranked = sorted(
    [(c, sum(v["sh"]) / len(v["sh"]), sum(v["ret"]) / len(v["ret"]),
      sorted(v["sh"])[len(v["sh"]) // 2],
      sum(1 for s in v["sh"] if s > 0.1), len(v["sh"]))
     for c, v in config_perf.items() if v["sh"]],
    key=lambda x: -x[1],
)

print(f"\n{'Config':<38s} {'Avg Sharpe':>10s} {'Avg Ret':>10s} {'Med Sharpe':>10s} {'Pos/Total':>10s}", flush=True)
print(f"{'-'*38} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)
for cfg, a_sh, a_ret, m_sh, pos, total in ranked:
    print(f"{cfg:<38s} {a_sh:>10.4f} {a_ret:>10.4f} {m_sh:>10.4f} {pos:>2d}/{total:<6d}", flush=True)

# --- Final judgment ---
improved = sum(1 for v in per_ticker.values()
               if v["sharpe_delta"] is not None and v["sharpe_delta"] > 0.01)
worsened = sum(1 for v in per_ticker.values()
               if v["sharpe_delta"] is not None and v["sharpe_delta"] < -0.01)
neutral = sum(1 for v in per_ticker.values()
              if v["sharpe_delta"] is not None and abs(v["sharpe_delta"]) <= 0.01)
nt = len(per_ticker)

print(f"\n{'='*100}", flush=True)
print("FINAL JUDGMENT", flush=True)
print(f"{'='*100}", flush=True)
print(f"  Tickers improved (Δ Sharpe > 0.01):  {improved:>2d}/{nt}")
print(f"  Tickers worsened  (Δ Sharpe < -0.01): {worsened:>2d}/{nt}")
print(f"  Tickers neutral   (|Δ| ≤ 0.01):       {neutral:>2d}/{nt}")

if ranked:
    print(f"\n  Best config (avg Sharpe):  {ranked[0][0]}  ({ranked[0][1]:.4f})")
    # Find DEFAULT rank
    for i, (c, *_) in enumerate(ranked):
        if c == "DEFAULT":
            print(f"  DEFAULT config rank:      #{i+1} / {len(ranked)}")
            break

print(f"\n  Results:      {results_path}")
print(f"  Analysis:     {OUT_DIR / 'robust_hmm_analysis.json'}")

# Save analysis
with open(OUT_DIR / "robust_hmm_analysis.json", "w") as f:
    json.dump({
        "per_ticker": per_ticker,
        "ranking": [{"config": c, "avg_sharpe": round(a, 6), "avg_return": round(r, 6),
                      "median_sharpe": round(m, 6), "positive": p, "total": t}
                     for c, a, r, m, p, t in ranked],
        "param_freq": dict(sorted(param_freq.items(), key=lambda x: -x[1])),
        "improved": improved, "worsened": worsened, "neutral": neutral,
        "total_tickers": nt,
    }, f, indent=2, allow_nan=False)
print(f"\nDone.", flush=True)
