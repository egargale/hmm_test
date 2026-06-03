#!/usr/bin/env python3
"""Run HMMMessina with the recommended configuration on all tickers.

Recommended config from HMMMessina_report.md:
  engine=messina, n_states=3, pca_variance=0.95, dwell_bars=0, hysteresis_delta=0.0

Also runs DEFAULT + hyst0 vs pca variants + hyst0 for full comparison.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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

# Configs to test (all with hysteresis_delta=0.0, dwell_bars=0)
CONFIGS = {
    "DEFAULT+hyst0": HMMMMessinaConfig(n_states=3, pca_variance=None),
    "RECOMMENDED":   HMMMMessinaConfig(n_states=3, pca_variance=0.95),
    "pca=0.99+hyst0": HMMMMessinaConfig(n_states=3, pca_variance=0.99),
    "n_states=5+hyst0": HMMMMessinaConfig(n_states=5, pca_variance=None),
    "n_states=auto+pca=0.95+hyst0": HMMMMessinaConfig(n_states="auto", pca_variance=0.95),
}

HYSTERESIS = 0.0
DWELL = 0
MIN_TRAIN = 252

csv_files = sorted(CSV_DIR.glob("*.csv"))
tickers = [f.stem for f in csv_files]
total_runs = len(CONFIGS) * len(csv_files)

print(f"Recommended config: n_states=3, pca_variance=0.95, dwell=0, hyst=0.0", file=sys.stderr)
print(f"Tickers ({len(tickers)}): {', '.join(tickers)}", file=sys.stderr)
print(f"Configs ({len(CONFIGS)}): {', '.join(CONFIGS.keys())}", file=sys.stderr)
print(f"Total runs: {total_runs}", file=sys.stderr)
print(file=sys.stderr)

all_results = []
run_count = 0

for csv_file in csv_files:
    ticker = csv_file.stem
    print(f"\n── {ticker} ──", flush=True)
    prices, ohlcv, source = load_prices(csv=str(csv_file))

    for label, config in CONFIGS.items():
        run_count += 1
        t0 = time.monotonic()
        try:
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
                "engine": "messina",
                "config": label,
                "n_states": str(config.n_states),
                "pca_variance": config.pca_variance,
                "dwell_bars": DWELL,
                "hysteresis_delta": HYSTERESIS,
                "n_states_resolved": output.engine_info.get("n_states"),
                "features_n": output.engine_info.get("features"),
                "degenerate": output.engine_info.get("degenerate", False),
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
                "engine": "messina",
                "config": label,
                "error": str(e)[:200],
                "wall_seconds": round(elapsed, 3),
            }

        all_results.append(rec)

        # Print inline
        if rec.get("error"):
            print(f"  [{run_count:2d}/{total_runs}] {label:28s} → ERROR: {rec['error']} [{elapsed:.1f}s]", flush=True)
        else:
            sharpe_s = f"{rec['sharpe']:.4f}" if rec['sharpe'] is not None else "NONE"
            ret_s = f"{rec['total_return']:.4f}" if rec['total_return'] is not None else "NONE"
            print(f"  [{run_count:2d}/{total_runs}] {label:28s} → sharpe={sharpe_s}  ret={ret_s}  trades={rec['n_trades']}  win={rec['win_rate']}  regime={rec['regime']}  [{elapsed:.1f}s]", flush=True)

# ── Save results ──
with open(OUT_DIR / "phase3_recommended.json", "w") as f:
    json.dump(all_results, f, indent=2, allow_nan=False)

# ── Pretty table ──
print(f"\n{'='*80}", file=sys.stderr)
print(f"RESULTS TABLE", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

headers = ["Ticker", "Config", "Sharpe", "Return", "Trades", "WinRate", "MaxDD", "Regime"]
print(f"\n  {' | '.join(h):30s}" % tuple(headers), file=sys.stderr)
print(f"  {'-'.join('-'*len(h) for h in headers)}", file=sys.stderr)

for csv_file in csv_files:
    t = csv_file.stem
    tk = [r for r in all_results if r["ticker"] == t]
    for r in tk:
        if r.get("error"):
            print(f"  {t:<8s} | {r['config']:<28s} | ERROR", file=sys.stderr)
        else:
            sh = f"{r['sharpe']:.4f}" if r['sharpe'] is not None else "NONE"
            rt = f"{r['total_return']:.4f}" if r['total_return'] is not None else "NONE"
            nt = str(r['n_trades'])
            wr = f"{r['win_rate']:.4f}" if r['win_rate'] is not None else "NONE"
            dd = f"{r['max_drawdown']:.4f}" if r['max_drawdown'] is not None else "NONE"
            rg = r['regime']
            print(f"  {t:<8s} | {r['config']:<28s} | {sh:>8s} | {rt:>8s} | {nt:>5s} | {wr:>7s} | {dd:>8s} | {rg}", file=sys.stderr)

# ── Summary ──
print(f"\n{'='*80}", file=sys.stderr)
print(f"SUMMARY", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

ok = [r for r in all_results if r.get("error") is None and r.get("sharpe") is not None]

for label in CONFIGS:
    rr = [r for r in ok if r["config"] == label]
    if not rr:
        continue
    avg_sh = sum(r["sharpe"] for r in rr) / len(rr)
    avg_ret = sum(r["total_return"] for r in rr) / len(rr)
    avg_trd = sum(r["n_trades"] for r in rr) / len(rr)
    wins = sum(1 for r in rr if r["sharpe"] > 0.1)
    total = len(rr)
    print(f"  {label:28s} | avg sharpe={avg_sh:+.4f}  avg return={avg_ret:+.4f}  avg trades={avg_trd:.0f}  sharpe>0.1={wins}/{total}", file=sys.stderr)

print(f"\nDone. Results saved to {OUT_DIR / 'phase3_recommended.json'}", file=sys.stderr)
