#!/usr/bin/env python3
"""Phase 2: HMMMessina parameter sweep — optimized.

Skips n_states=2 (shape bug with 3-state names).
Runs one ticker per invocation for reliable timeout handling.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run
from hmm_futures_analysis.utils.logging_config import suppress_stdout_logging

suppress_stdout_logging()

ticker = sys.argv[1] if len(sys.argv) > 1 else "ALL"
print(f"SWEEP: ticker={ticker}", flush=True)

CSV_DIR = Path("test_data/eval-results")
OUT_DIR = Path("test_data/eval-results/messina_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter grid — skip n_states=2 (always errors with shape (2,) vs (3,) broadcast)
N_STATES_OPTIONS = [3, 4, 5, "auto"]
PCA_OPTIONS = [None, 0.95, 0.99]
DWELL_OPTIONS = [0, "auto", 2, 5]
HYST_OPTIONS = [0.0, 0.1, 0.05, 0.2]

total_combos = len(N_STATES_OPTIONS) * len(PCA_OPTIONS) * len(DWELL_OPTIONS) * len(HYST_OPTIONS)

if ticker == "ALL":
    csv_files = sorted(CSV_DIR.glob("*.csv"))
else:
    csv_path = CSV_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)
    csv_files = [csv_path]

all_results = []

for csv_file in csv_files:
    tkr = csv_file.stem
    print(f"\n── {tkr} ({total_combos} combos) ──", flush=True)
    prices, ohlcv, source = load_prices(csv=str(csv_file))

    run_count = 0
    for n_states, pca_var, dwell, hyst in itertools.product(
        N_STATES_OPTIONS, PCA_OPTIONS, DWELL_OPTIONS, HYST_OPTIONS
    ):
        config = HMMMMessinaConfig(n_states=n_states, pca_variance=pca_var)
        run_count += 1

        # Build compact label
        ns = f"ns={n_states}"
        pc = f"pca={pca_var}" if pca_var is not None else "pca=None"
        dw = f"dwell={dwell}"
        hy = f"hyst={hyst}"
        label = f"{ns}|{pc}|{dw}|{hy}"

        try:
            t0 = time.monotonic()
            output = pipeline_run(
                prices=prices,
                source=tkr,
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
                "ticker": tkr,
                "engine": "messina",
                "config": label,
                "n_states": str(n_states),
                "pca_variance": pca_var,
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
            rec = {
                "ticker": tkr,
                "engine": "messina",
                "config": label,
                "n_states": str(n_states),
                "pca_variance": pca_var,
                "dwell_bars": dwell,
                "hysteresis_delta": hyst,
                "error": str(e),
                "sharpe": None,
                "n_trades": None,
                "win_rate": None,
                "profit_factor": None,
                "total_return": None,
                "max_drawdown": None,
                "signal": None,
                "regime": None,
                "wall_seconds": None,
            }

        all_results.append(rec)

        # Oneline progress
        sharpe_s = f"{rec['sharpe']:.4f}" if rec['sharpe'] is not None else "ERR"
        ret_s = f"{rec['total_return']:.4f}" if rec['total_return'] is not None else "ERR"
        trd_s = f"{rec['n_trades']}" if rec['n_trades'] is not None else "ERR"
        print(f"  [{run_count:3d}/{total_combos}] {label:42s} "
              f"sharpe={sharpe_s}  ret={ret_s}  trades={trd_s}", flush=True)

    # Save per-ticker results
    outfile = OUT_DIR / f"phase2_{tkr}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, allow_nan=False)
    print(f"  → Saved {outfile}", flush=True)
    print(f"  → Done {tkr}: {len(all_results)} results", flush=True)

print(f"\n✓ Complete. All results in {OUT_DIR}/", flush=True)
