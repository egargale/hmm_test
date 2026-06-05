#!/usr/bin/env python3
"""Validate regime detection across all engines × CSV files.
Runs engines one at a time, saves results to JSON."""
import sys, os, json, time, warnings, traceback
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import (
    ThresholdConfig, HMMGenericConfig, HMMMMessinaConfig,
    RobustHMMConfig, FSHMMConfig,
)
from hmm_futures_analysis.regime.pipeline import run as pipeline_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, "test_data", "eval-results")
OUTPUT = os.path.join(ROOT, "test_data", "regime_validation.json")

ENGINES = [
    ("threshold", ThresholdConfig()),
    ("hmm", HMMGenericConfig()),
    ("messina", HMMMMessinaConfig()),
    ("robust_hmm", RobustHMMConfig()),
    ("fshmm", FSHMMConfig()),
]

csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
all_results = {}
all_issues = []

for csv_file in csv_files:
    ticker = csv_file.replace(".csv", "")
    path = os.path.join(CSV_DIR, csv_file)
    try:
        prices, ohlcv, source = load_prices(csv=path)
        n_bars = len(prices)
    except Exception as e:
        print(f"ERROR loading {csv_file}: {e}")
        all_issues.append(f"{ticker}: LOAD ERROR: {e}")
        continue

    print(f"\n{'='*80}")
    print(f"{ticker}: {n_bars} bars | {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"{'='*80}")

    for eng_name, config in ENGINES:
        label = f"{ticker}/{eng_name}"
        t0 = time.time()
        try:
            result = pipeline_run(
                prices, source=source, engine_config=config,
                ohlcv=ohlcv, profile=False
            )
            r = result._asdict()
            elapsed = time.time() - t0

            regime_name = r["current_regime"]["name"]
            regime_idx = r["current_regime"]["index"]
            counts = r["regime_counts"]
            total = sum(counts.values())
            signal = r["signal"]
            verdict = r["verdict"]["verdict"]
            confidence = r["verdict"]["confidence"]
            wf = r["walk_forward"]
            eng_info = r["engine_info"]
            persist = r["persistence_diagonal"]

            issues = []

            # 1. Regime balance
            if total > 0:
                for state, count in counts.items():
                    frac = count / total
                    if frac == 0.0:
                        issues.append(f"ZERO {state} bars (regime never visited)")
                    elif frac > 0.95:
                        issues.append(f"{state} dominates at {frac:.1%} (degenerate)")

            # 2. Walk-forward trades
            wf_sharpe = wf.get("sharpe")
            wf_trades = wf.get("n_trades", 0)
            if wf_trades == 0 and n_bars > 300:
                issues.append("0 walk-forward trades on >300 bars dataset")

            # 3. Degenerate fit
            if eng_info.get("degenerate_fit"):
                issues.append(f"DEGENERATE: {eng_info.get('degenerate_caveat', '')[:80]}")

            # 4. Low data warning
            if eng_info.get("low_data_warning"):
                issues.append(f"LOW DATA: {eng_info.get('low_data_caveat', '')[:80]}")

            # 5. Signal range
            if abs(signal) > 1.0:
                issues.append(f"signal out of range [-1,1]: {signal:.4f}")

            # 6. Valid regime index
            if regime_idx not in (0, 1, 2):
                issues.append(f"invalid regime index: {regime_idx}")

            # 7. Persistence sanity
            for state, pval in persist.items():
                if pval is not None and not (0 <= pval <= 1):
                    issues.append(f"persistence {state} out of [0,1]: {pval}")

            # 8. Transition matrix row sums
            tm = np.array(r["transition_matrix"])
            row_sums = tm.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                issues.append(f"transition matrix rows don't sum to 1: {row_sums}")

            status = "OK" if not issues else "ISSUES"
            pct = {k: f"{v/total:.1%}" if total > 0 else "0%" for k, v in counts.items()}

            print(f"  {eng_name:12s} | {elapsed:5.1f}s | regime={regime_name:8s} | "
                  f"bear={counts['bear']:>5d}({pct['bear']:>5s}) sidew={counts['sideways']:>5d}({pct['sideways']:>5s}) bull={counts['bull']:>5d}({pct['bull']:>5s}) | "
                  f"sig={signal:+.3f} | verd={verdict:16s} conf={confidence:.2f} | "
                  f"wf_sharpe={str(wf_sharpe):>8s} wf_trades={wf_trades:>3d} | {status}")

            if issues:
                for iss in issues:
                    print(f"    WARNING: {iss}")
                    all_issues.append(f"{label}: {iss}")

            all_results[label] = {
                "regime": regime_name,
                "signal": signal,
                "verdict": verdict,
                "confidence": confidence,
                "counts": counts,
                "wf_sharpe": wf_sharpe,
                "wf_trades": wf_trades,
                "persistence": persist,
                "engine_info": {k: v for k, v in eng_info.items()
                               if k not in ("degenerate_caveat", "low_data_caveat")},
                "elapsed_s": round(elapsed, 2),
                "issues": issues,
                "status": status,
            }

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {eng_name:12s} | {elapsed:5.1f}s | ERROR: {e}")
            all_issues.append(f"{label}: RUNTIME: {e}")
            traceback.print_exc()
            all_results[label] = {"error": str(e), "elapsed_s": round(elapsed, 2)}

# Save
with open(OUTPUT, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Summary
total_runs = len(csv_files) * len(ENGINES)
ok_count = sum(1 for v in all_results.values() if v.get("status") == "OK")
issue_count = sum(1 for v in all_results.values() if v.get("status") == "ISSUES")
error_count = sum(1 for v in all_results.values() if "error" in v)

print(f"\n{'='*80}")
print(f"SUMMARY: {total_runs} total runs")
print(f"  OK:     {ok_count}")
print(f"  ISSUES: {issue_count}")
print(f"  ERRORS: {error_count}")
print(f"  Saved:  {OUTPUT}")

if all_issues:
    print(f"\nALL ISSUES ({len(all_issues)}):")
    for i, iss in enumerate(all_issues, 1):
        print(f"  {i}. {iss}")
