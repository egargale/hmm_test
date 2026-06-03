# HMMMessina Parameter Sensitivity & Evaluation Report

**Date:** 2026-06-03
**Engine:** HMMMessina (HMM with 19 Messina-style features)
**Data:** 5 CSV files in `test_data/eval-results/` — 0700_HK, BTC, CRM, KO, SPY

---

## Executive Summary

The default HMMMessina configuration is **broken for trading**: `hysteresis_delta=0.1` kills nearly all trades (1 trade/ticker). With `hysteresis_delta=0.0` and `pca_variance=0.95`, the engine delivers a **dramatic turnaround** — average Sharpe goes from **-0.31 → +0.32** and average return from **-43% → +92%** across 4 tickers.

| Metric | Default | Recommended (`pca=0.95`, `hyst=0.0`) | Improvement |
|---|---|---|---|
| **Avg Sharpe** | -0.31 | **+0.32** | +0.63 |
| **Avg Total Return** | -43% | **+92%** | +135 pp |
| **Avg Trades** | 1 | **21** | 20× more |

---

## Scope & Methodology

### Tickers Evaluated

| File | Rows | Viable? | Notes |
|---|---|---|---|
| `0700_HK.csv` | ~2500 | ✅ | Tencent |
| `BTC.csv` | ~1600 | ✅ | Bitcoin |
| `CRM.csv` | ~2600 | ✅ | Salesforce |
| `KO.csv` | ~2600 | ✅ | Coca-Cola |
| `SPY.csv` | 124 | ❌ | Too short for `min_train=252` |

SPY was excluded from all analysis (124 rows vs 252-bar minimum training window → 0 trades for all configurations).

### What Was Tested

**Phase 1** — Default HMMMessina on all 5 tickers:
- `n_states=3, pca_variance=None, dwell_bars=0, hysteresis_delta=0.1`

**Phase 2** — Surgical sensitivity sweep (12 runs per ticker = 48 total):
- Vary **hysteresis_delta**: 0.0, 0.1, 0.05
- Vary **n_states**: 3, 4, 5, auto
- Vary **pca_variance**: None, 0.95, 0.99
- Vary **dwell_bars**: 0, auto, 2, 5
- One "best guess" combo: `n_states=auto, pca=0.95, dwell=0, hyst=0.0`

A full 192-combo grid search was attempted but each walk-forward run takes ~25s due to HMM refitting at every backtest step — 5×192×25s = ~7 hours even with parallelism. The surgical 1-at-a-time approach was the practical alternative.

---

## Phase 1 — Default Results

The default HMMMessina produces exactly **1 trade per ticker**, killing all signal:

| Ticker | Sharpe | Total Return | Trades | Win Rate | Max DD |
|---|---|---|---|---|---|
| 0700_HK | -0.43 | -83.3% | 1 | 0% | -88.6% |
| BTC | +0.36 | +66.9% | 1 | 100% | -66.7% |
| CRM | -0.50 | -87.1% | 1 | 0% | -91.4% |
| KO | -0.69 | -67.9% | 1 | 0% | -70.6% |
| **Avg** | **-0.31** | **-42.9%** | **1** | — | — |

The single trade is a short-only position the model enters and stays in — the hysteresis filter never allows a flip back. The one exception (BTC) shows a short-then-long pattern but also only 1 trade.

**Root cause:** `hysteresis_delta=0.1` requires the posterior probability of the new regime to exceed the old regime by 0.1, which is too high a hurdle for the HMM's smoothed posteriors.

---

## Phase 2 — Sensitivity Sweep Results

### 0700_HK (Tencent)

| Config | Sharpe | Return | Trades |
|---|---|---|---|
| DEFAULT (hyst=0.1) | -0.43 | -83.3% | 1 |
| DEFAULT+hyst0 | +0.06 | +4.7% | 11 |
| n_states=4+hyst0 | -0.38 | -71.2% | 20 |
| n_states=5+hyst0 | -0.55 | -59.8% | 19 |
| n_states=auto+hyst0 | -0.10 | -34.2% | 27 |
| **pca=0.95+hyst0** | **+0.03** | **-2.1%** | **15** |
| **pca=0.99+hyst0 ★** | **+0.51** | **+209.8%** | **20** |
| dwell=auto\|hyst=0.0 | +0.06 | +4.7% | 11 |
| dwell=2\|hyst=0.0 | -0.09 | -23.7% | 11 |
| dwell=5\|hyst=0.0 | -0.12 | -28.5% | 11 |
| hyst=0.05 | -0.43 | -83.3% | 1 |
| best_guess | -0.37 | -72.5% | 11 |

**Best:** `pca=0.99+hyst0` — Sharpe +0.51, Return +210%
**Insight:** PCA is transformative for 0700_HK. The Messina features have high multicollinearity and PCA untangles them.

### BTC (Bitcoin)

| Config | Sharpe | Return | Trades |
|---|---|---|---|
| DEFAULT (hyst=0.1) | **+0.36** | +66.9% | 1 |
| DEFAULT+hyst0 | -0.56 | -76.6% | 14 |
| n_states=4+hyst0 | -0.98 | -88.8% | 19 |
| n_states=5+hyst0 | -0.51 | -60.4% | 26 |
| n_states=auto+hyst0 | -0.58 | -79.4% | 18 |
| **pca=0.95+hyst0 ★** | **+0.67** | **+228.1%** | **26** |
| pca=0.99+hyst0 | +0.55 | +153.9% | 19 |
| dwell=auto\|hyst=0.0 | -0.56 | -76.6% | 14 |
| dwell=2\|hyst=0.0 | -0.60 | -78.4% | 14 |
| dwell=5\|hyst=0.0 | -0.52 | -74.2% | 14 |
| hyst=0.05 | +0.36 | +66.9% | 1 |
| best_guess | -0.35 | -54.2% | 25 |

**Best:** `pca=0.95+hyst0` — Sharpe +0.67, Return +228%
**Insight:** BTC is unique — default actually makes money (+67% return) with just 1 trade. But PCA + hyst=0.0 blows it away: 26 trades, +228% return. PCA whitening captures the regime structure.

### CRM (Salesforce)

| Config | Sharpe | Return | Trades |
|---|---|---|---|
| DEFAULT (hyst=0.1) | -0.50 | -87.1% | 1 |
| DEFAULT+hyst0 | -0.23 | -36.1% | 15 |
| n_states=4+hyst0 | -0.56 | -88.5% | 8 |
| n_states=5+hyst0 | -0.54 | -75.3% | 24 |
| n_states=auto+hyst0 | -0.49 | -74.1% | 24 |
| **pca=0.95+hyst0** | **+0.33** | **+96.9%** | **27** |
| pca=0.99+hyst0 | -0.05 | -34.6% | 24 |
| dwell=auto\|hyst=0.0 | -0.23 | -36.1% | 15 |
| dwell=2\|hyst=0.0 | -0.24 | -37.1% | 15 |
| dwell=5\|hyst=0.0 | -0.19 | -30.7% | 15 |
| hyst=0.05 | -0.50 | -87.1% | 1 |
| **best_guess ★** | **+0.46** | **+172.0%** | **23** |

**Best:** `best_guess` (n_states=auto, pca=0.95, hyst=0.0) — Sharpe +0.46, Return +172%
**Close second:** `pca=0.95+hyst0` — Sharpe +0.33, Return +97%

### KO (Coca-Cola)

| Config | Sharpe | Return | Trades |
|---|---|---|---|
| DEFAULT (hyst=0.1) | -0.69 | -67.9% | 1 |
| DEFAULT+hyst0 | -0.13 | -6.0% | 14 |
| n_states=4+hyst0 | -0.30 | -33.8% | 11 |
| n_states=5+hyst0 | +0.14 | +30.7% | 28 |
| n_states=auto+hyst0 | -0.47 | -43.5% | 18 |
| **pca=0.95+hyst0 ★** | **+0.24** | **+44.8%** | **16** |
| pca=0.99+hyst0 | +0.02 | +10.1% | 20 |
| dwell=auto\|hyst=0.0 | -0.13 | -6.0% | 14 |
| dwell=2\|hyst=0.0 | -0.11 | -2.6% | 14 |
| dwell=5\|hyst=0.0 | -0.25 | -19.0% | 14 |
| hyst=0.05 | -0.69 | -67.9% | 1 |
| best_guess | +0.02 | +15.1% | 22 |

**Best:** `pca=0.95+hyst0` — Sharpe +0.24, Return +45%
**Runner-up:** `n_states=5+hyst0` — Sharpe +0.14, Return +31% (most trades: 28)

---

## Per-Parameter Analysis

### 1. Hysteresis Delta — 🟥 CRITICAL (Default is Harmful)

| Setting | Avg Trades | Avg Sharpe | Avg Return |
|---|---|---|---|
| **hyst=0.1 (default)** | **1** | **-0.31** | **-43%** |
| hyst=0.05 | 1 | -0.31 | -43% |
| **hyst=0.0** | **18** | **-0.21** | **-29%** |
| hyst=0.0 + PCA 0.95 | **21** | **+0.32** | **+92%** |

- `hyst=0.1` kills all trades: 1 trade/ticker, no ability to flip positions
- Even `hyst=0.05` is as restrictive as `hyst=0.1`
- `hyst=0.0` enables active trading (14-28 trades/ticker)
- When combined with PCA, `hyst=0.0` unlocks positive returns

**Verdict:** The default `hysteresis_delta=0.1` is actively harmful. **Recommended: 0.0.**

### 2. PCA Whitening — 🟢 Most Impactful Positive Parameter

| Setting | Wins (best Sharpe) | Avg Sharpe |
|---|---|---|
| **pca=0.95** | **2/4** (BTC, KO) | **+0.32** |
| pca=0.99 | 1/4 (0700_HK) | +0.26 |
| pca=None | 1/4 (CRM via best_guess) | -0.21 |

- `pca_variance=0.95` is the single most impactful positive parameter
- PCA 0.95 delivers positive returns on 3 of 4 tickers (vs 0 of 4 for default)
- PCA 0.99 is also beneficial but slightly weaker
- PCA helps because the 19 Messina features have multicollinearity

**Verdict:** **Recommended: pca_variance=0.95.**

### 3. Number of States — ⚠️ Context-Dependent

| Setting | Best on | Notes |
|---|---|---|
| n_states=3 | Baseline | Competitive with hyst=0.0 |
| n_states=4 | None | Generally worse |
| n_states=5 | KO (Sharpe +0.14) | Better on stable instruments |
| n_states=auto | CRM (Sharpe +0.46) | Works well + PCA for CRM |

- `n_states=3` is the most robust default
- `n_states=5` helps on KO (defensive stock)
- `n_states=auto` + PCA gives best CRM result (high-volatility stock)
- `n_states=4` never wins on any ticker

**Verdict:** Keep `n_states=3` as default. Consider `n_states=5` for stable/defensive instruments.

### 4. Dwell Bars — 🟢 Minimal Impact

| Setting | Avg Sharpe | Δ from dwell=0 |
|---|---|---|
| dwell=0 | -0.21 | baseline |
| dwell=auto | -0.21 | 0.00 |
| dwell=2 | -0.22 | -0.01 |
| dwell=5 | -0.23 | -0.02 |

- `dwell=auto` resolves to 0 for HMMMessina
- `dwell>0` slightly hurts returns without improving drawdown
- When hyst=0.0, dwell adds no value

**Verdict:** Keep `dwell_bars=0` (disabled).

---

## Best Combination per Ticker

| Ticker | Best Config | Sharpe | Return | Trades |
|---|---|---|---|---|
| 0700_HK | pca=0.99, n_states=3, dwell=0, hyst=0.0 | **+0.51** | **+210%** | 20 |
| BTC | pca=0.95, n_states=3, dwell=0, hyst=0.0 | **+0.67** | **+228%** | 26 |
| CRM | n_states=auto, pca=0.95, dwell=0, hyst=0.0 | **+0.46** | **+172%** | 23 |
| KO | pca=0.95, n_states=3, dwell=0, hyst=0.0 | **+0.24** | **+45%** | 16 |

**Universal winning pattern:** `hyst=0.0` + some form of PCA (0.95 or 0.99). `n_states=3` works well for most.

---

## Recommended Configuration

```python
from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig

config = HMMMMessinaConfig(
    n_states=3,
    pca_variance=0.95,
)
```

CLI usage:
```bash
python -m hmm_futures_analysis.cli --csv data.csv --engine messina \
    --n-states 3 --hysteresis 0.0 --dwell-bars 0
# PCA not exposed via CLI for messina engine currently; would need CLI update
```

### Comparison: Default vs Recommended

| Metric | Default (avg) | Recommended (avg) | Change |
|---|---|---|---|
| **Sharpe** | -0.31 | **+0.32** | +0.63 |
| **Total Return** | -43% | **+92%** | +135 pp |
| **Trades** | 1 | **21** | 20× |
| **Max Drawdown** | ~-79% | N/A (not tracked in sensitivity) | — |

---

## Open Questions & Next Steps

1. **Drawdown tracking:** The sensitivity sweep didn't capture `max_drawdown` for all variants — the `DEFAULT+hyst0` runs return `max_drawdown: None` from the walk-forward backtest. Investigate why.

2. **Walk-forward stability:** PCA with `pca=0.95` means the HMM sees different feature sets at each walk-forward step (PCA refits on expanding window). This could cause regime instability. Validate by comparing PCA vs non-PCA regime sequences.

3. **Per-instrument tuning:** The best n_states varies by ticker (3 for BTC/0700_HK, 5 for KO, auto for CRM). Consider ticker-specific config profiles.

4. **hysteresis = 0.01 or 0.02:** The jump from 0.1 → 0.0 is extreme. A small non-zero value like 0.01 might reduce whipsaw without killing all trades. Not tested.

5. **Feature-level analysis:** The 19 Messina features have known multicollinearity. With PCA=0.95, which features survive the whitening? This could guide feature selection improvements.

---

## Raw Results Location

All JSON results saved to `test_data/eval-results/messina_sweep/`:
- `phase1_defaults.json` — Default HMMMessina on all 5 CSVs
- `phase2_sensitivity.json` — 48 sensitivity sweep results (12 runs × 4 tickers)
