#!/usr/bin/env python3
"""Run only the missing pipeline combinations for the HMM sweep."""

import pickle, time
from pathlib import Path
from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run

SWEEP_DIR = Path("test_data/eval-results/hmm_sweep")
CSV_DIR = Path("test_data/eval-results")
CACHE_DIR = SWEEP_DIR / "pipeline_cache"
TICKERS = ["0700_HK", "BTC", "CRM", "KO", "SPY"]

def cache_key(t, ns, pca, dw, hy):
    ns_s = f"ns{ns}" if not isinstance(ns, str) else f"ns{ns}"
    pc_s = "pcaNone" if pca is None else f"pca{str(pca).replace('.','_')}"
    dw_s = f"dw{dw}"
    hy_s = f"hy{str(hy).replace('.','_')}"
    return f"{t}__{ns_s}_{pc_s}_{dw_s}_{hy_s}"

def already_cached(t, ns, pca, dw, hy):
    return (CACHE_DIR / f"{cache_key(t, ns, pca, dw, hy)}.pkl").exists()

def run_one(t, ns, pca, dw, hy):
    ck = cache_key(t, ns, pca, dw, hy)
    print(f"  {ck} ...", end=" ", flush=True)
    csv_path = CSV_DIR / f"{t}.csv"
    config = HMMGenericConfig(n_states=ns, pca_variance=pca)
    prices, ohlcv, source = load_prices(csv=str(csv_path))
    t0 = time.monotonic()
    output = pipeline_run(prices=prices, source=t, engine_config=config,
                          min_train=252, ohlcv=ohlcv, dwell_bars=dw, hysteresis_delta=hy)
    elapsed = time.monotonic() - t0
    with open(CACHE_DIR / f"{ck}.pkl", "wb") as f:
        pickle.dump({"output": output, "elapsed": elapsed}, f)
    print(f"{elapsed:.0f}s sharpe={output.walk_forward['sharpe']:.4f} trades={output.walk_forward['n_trades']}")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Missing combos:
# 1. PCA: n_states=3, pca=[0.90, 0.95, 0.99], dwell=0, hyst=0.0
print("=== PCA combos ===")
pca_vals = [0.90, 0.95, 0.99]
for t in TICKERS:
    for pca in pca_vals:
        ck = cache_key(t, 3, pca, 0, 0.0)
        if already_cached(t, 3, pca, 0, 0.0):
            print(f"  {ck} [cached]")
        else:
            run_one(t, 3, pca, 0, 0.0)

# 2. SPY auto (missing)
print("\n=== SPY auto ===")
if not already_cached("SPY", "auto", None, 0, 0.0):
    run_one("SPY", "auto", None, 0, 0.0)
else:
    print("  SPY auto [cached]")

# 3. Dwell: n_states=3, pca=None, dwell=[2,3,5], hyst=0.0
print("\n=== dwell combos ===")
dwell_vals = [2, 3, 5]
for t in TICKERS:
    for dw in dwell_vals:
        if already_cached(t, 3, None, dw, 0.0):
            print(f"  {cache_key(t, 3, None, dw, 0.0)} [cached]")
        else:
            run_one(t, 3, None, dw, 0.0)

# 4. Hysteresis: n_states=3, pca=None, dwell=0, hyst=[0.05, 0.1, 0.2]
print("\n=== hysteresis combos ===")
hyst_vals = [0.05, 0.1, 0.2]
for t in TICKERS:
    for hy in hyst_vals:
        if already_cached(t, 3, None, 0, hy):
            print(f"  {cache_key(t, 3, None, 0, hy)} [cached]")
        else:
            run_one(t, 3, None, 0, hy)

print("\nDone!")
