# USAGE.md — HMM Regime Detection

Complete reference for the `hmm-regime-detection` CLI and its five analysis engines.

---

## Quick Start

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Simplest possible run — threshold engine, live ticker
python -m hmm_futures_analysis.cli --ticker SPY --json

# Or use the shell wrapper
./scripts/run.sh --ticker SPY --json
```

> **Important:** Always activate the venv (`.venv/bin/activate`) or use
> `.venv/bin/python -m` directly. The system Python will not have the
> `ta` library, causing HMM engines to run with degraded features.

---

## Data Input

| Flag | Description |
|---|---|
| `--csv PATH` | Path to a CSV file with OHLCV data. Auto-detects column names and date format. |
| `--ticker SYMBOL` | yfinance ticker symbol. Downloads daily data automatically. |

One of `--csv` or `--ticker` is required. They are mutually exclusive.

### CSV format

The CSV auto-detector handles most common layouts. It looks for columns named
`Close`, `Adj Close`, `Price`, `Value`, etc., and a date-like column for the
index. OHLCV columns (`Open`, `High`, `Low`, `Close`, `Volume`) are used by
HMM engines for feature computation.

For best results, provide a CSV with at least: `Date, Open, High, Low, Close, Volume`.

---

## Engines

### 1. `threshold` — Simple Rolling Return

**Default engine.** Fast, needs only close prices. No HMM fitting.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine threshold --json
```

#### How it works

1. Computes daily returns from close prices.
2. Calculates rolling `window`-bar cumulative return.
3. Labels each bar:
   - **Bull** (2) — cumulative return > `+threshold`
   - **Bear** (0) — cumulative return < `−threshold`
   - **Sideways** (1) — in between
4. Builds a Markov transition matrix from the regime sequence.

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--window` | 20 | Rolling window (bars) for cumulative return |
| `--threshold` | 0.05 | Return threshold for bull/bear classification (5%) |

#### Strengths

- **Speed** — instant, no iterative fitting.
- **Robustness** — works with close-only data (no OHLCV needed).
- **Interpretability** — simple threshold logic, no hidden state ambiguity.

#### Weaknesses

- **Whipsaw** — on noisy data, the rolling return oscillates around zero,
  generating excessive regime switches (100+ trades in 10 years is common).
- **Lagging** — the rolling window is a lagging indicator; regime changes are
  detected late.
- **No posteriors** — produces hard labels, not probability distributions.

#### When to use

- Quick sanity check or baseline comparison.
- Data with only close prices (no OHLCV).
- Very short histories (< 500 bars) where HMM fitting is unreliable.

---

### 2. `messina` — HMM + 19 Curated Features

HMM engine using 19 hand-crafted "Messina" features designed for regime
detection. Requires OHLCV data.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine messina --json
```

#### The 19 Messina Features

| # | Feature | Description |
|---|---|---|
| 1 | `log_ret` | Log return |
| 2 | `sma_200` | 200-bar simple moving average |
| 3 | `sma_13` | 13-bar simple moving average |
| 4 | `atr_20` | 20-bar average true range |
| 5 | `adx_14` | 14-bar average directional index |
| 6 | `adx_inflection` | ADX direction change flag |
| 7 | `di_plus_14` | +DI (14-bar) |
| 8 | `di_minus_14` | −DI (14-bar) |
| 9 | `di_spread` | +DI − −DI |
| 10 | `vstop` | Volatility stop (parabolic ATR) |
| 11 | `vstop_trend` | Vstop trend direction (+1/−1) |
| 12 | `vstop_interaction` | Price vs vstop interaction |
| 13 | `price_sma200_ratio` | Close / SMA(200) |
| 14 | `price_vstop_ratio` | Close / Vstop |
| 15 | `price_vstop_gap_atr` | (Close − Vstop) / ATR |
| 16 | `sma200_distance_atr` | (Close − SMA200) / ATR |
| 17 | `volume_ratio` | Volume / 20-bar volume SMA |
| 18 | `true_range_pct` | True range / Close |
| 19 | `kdj_j` | KDJ stochastic J-value |

These features use **Wilder's smoothing** (exponential with α = 1/period),
not simple SMA. This produces smoother, more persistent signals that reduce
HMM state-switching noise.

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n-states` | 3 | HMM states: integer ≥ 2, or `auto` (BIC selection, 2–6) |
| `--min-train` | 252 | Warmup bars before walk-forward trading begins |

#### Strengths

- **Curated features** — designed specifically for regime detection, not generic.
- **Low trade count** — sticky regimes (often < 20 trades in 10 years).
- **Moderate data requirement** — 19 features avoids overfitting on short histories.
- **Wilder's smoothing** — reduces noise compared to SMA-based features.

#### Weaknesses

- **Feature rigidity** — the 19 features are fixed; can't adapt to new assets.
- **Slow convergence** — Wilder's smoothing takes ~14× the period to stabilise.
- **No feature selection** — all 19 features are always used, even if some are noise.

#### When to use

- General-purpose regime detection on established equities.
- Medium-length histories (500+ bars).
- When you want stable, low-turnover regime signals.

---

### 3. `hmm` — HMM + ~50 Generic Features

Full-featured HMM engine using the `ta` technical analysis library plus
hand-computed features. Requires OHLCV data.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine hmm --json
```

#### Feature Categories

The engine computes features from the OHLCV DataFrame via `add_features()`.
The exact count depends on data availability (intraday vs daily). On daily
equity data, approximately 50 features are produced:

| Category | Features | Examples |
|---|---|---|
| **Returns** | 2 | `log_ret`, `simple_ret` |
| **Moving averages** | ~8 | `sma_20`, `ema_20`, `sma_ratio_50_200` |
| **Volatility** | ~5 | `atr_14`, `bb_width_20`, `bb_position_20`, historical volatility |
| **Momentum** | ~5 | `rsi_14`, `roc_10`, `macd`, `macd_signal`, `stoch_k` |
| **Volume** | ~4 | `volume_sma_20`, `volume_ratio_20`, `obv`, `vwap` |
| **Trend** | ~5 | `adx_14`, `adx_plus_14`, `adx_minus_14`, `cci_20`, `dpo_20` |
| **Price patterns** | ~6 | `bb_upper_20`, `bb_middle_20`, `bb_lower_20`, Keltner, Donchian |
| **Enhanced momentum** | ~5 | `williams_r_14`, `cci_20`, `mfi_14`, `mtm_10`, `roc_10` |
| **Enhanced volatility** | ~4 | `chaikin_vol_20`, historical volatility, Keltner, Donchian |
| **Enhanced trend** | ~5 | `tma_*`, `wma_*`, `hma_*`, `aroon_*`, `di_*`, `adx_*` |
| **Volume (enhanced)** | ~3 | `adl`, `vpt`, `eom_*`, `volume_roc_*` |
| **Calendar** | ~6 | `day_of_week_sin/cos`, `day_of_month`, `is_month_end`, `is_quarter_end`, `month_sin/cos` |

> **Note:** If the `ta` library is not installed, features that depend on it
> (moving averages, Bollinger bands, RSI, MACD, ADX, etc.) are skipped. The
> engine still runs with the hand-computed subset (~25 features), but output
> quality degrades significantly. Always verify `ta` is importable in your
> environment.

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n-states` | 3 | HMM states: integer ≥ 2, or `auto` (BIC selection, 2–6) |
| `--min-train` | 252 | Warmup bars before walk-forward trading begins |

#### Strengths

- **Comprehensive feature set** — covers all major technical indicator families.
- **BIC auto-selection** — `--n-states auto` tries 2–6 states and picks the best.
- **Full `ta` library** — when installed, gets ~50 features from scipy-grade
  indicator implementations.

#### Weaknesses

- **Overfitting risk** — 50 features on short histories (< 1000 bars) leads to
  severe overfitting. The model may collapse to a 2-regime model.
- **No feature selection** — all features are always used, even irrelevant ones.
- **Slow** — fitting 50-dimensional Gaussian mixtures is computationally
  expensive, especially with walk-forward re-fitting.

#### When to use

- Long histories (2000+ bars) where the feature count is supported by data.
- When you want the most features possible and don't mind the tuning burden.
- As a baseline for comparison with `fshmm` (which adds feature selection).

---

### 4. `robust_hmm` — Outlier-Resistant HMM

Same feature set as `hmm` (~50 generic features), but uses robust estimation
for the Gaussian emission parameters. Requires OHLCV data.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine robust_hmm --json
```

#### Robust Methods

| Method | Flag | Description |
|---|---|---|
| **Huber IRLS** | `--robust-method huber` (default) | Iteratively reweighted least squares with Huber loss. Downweights outliers in the mean/covariance estimation. |
| **MinCovDet** | `--robust-method mcd` | Minimum Covariance Determinant from scikit-learn. Fast approximate robust covariance. |

The robust method is used to compute the initial emission parameters before
EM refinement. This prevents flash crashes, earnings gaps, and other outliers
from distorting the regime model.

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n-states` | 3 | HMM states: integer ≥ 2, or `auto` (BIC selection, 2–6) |
| `--min-train` | 252 | Warmup bars before walk-forward trading begins |
| `--robust-method` | `huber` | Robust estimator: `huber` or `mcd` |

#### Strengths

- **Outlier resistance** — flash crashes and earnings gaps don't corrupt the model.
- **Same features as `hmm`** — full technical indicator coverage.
- **Two methods** — Huber is faster; MCD is more aggressive at outlier rejection.

#### Weaknesses

- **Same overfitting risk as `hmm`** — 50 features with no feature selection.
- **Over-robustness** — on clean data, robust estimation can suppress genuine
  regime signals, causing the model to "lock" into one regime.
- **Extra compute** — robust fitting adds overhead vs. standard `hmm`.

#### When to use

- Assets with frequent outliers (crypto, small-caps, earnings-driven stocks).
- When `hmm` produces degenerate fits (all one regime, NaN posteriors).

---

### 5. `fshmm` — Feature Saliency HMM

Same feature set as `hmm` (~50 generic features), but learns per-feature
**saliency weights** during EM training. Features below the saliency threshold
are automatically masked as irrelevant. Requires OHLCV data.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine fshmm --json
```

#### How Feature Saliency Works

Based on Adams et al. (2016). During EM training, each feature k gets a
saliency weight ρ_k ∈ [0, 1]:

- ρ_k close to **1** — feature is informative for regime discrimination.
- ρ_k close to **0** — feature is noise and should be ignored.

After training, features with ρ_k < `saliency_threshold` are pruned. The
output includes the full saliency vector and the selected feature names.

```bash
# Lower threshold = more features used
python -m hmm_futures_analysis.cli --ticker SPY --engine fshmm --saliency-threshold 0.3 --json

# Save saliency weights for analysis
python -m hmm_futures_analysis.cli --ticker SPY --engine fshmm --saliency-output saliency.csv --json
```

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n-states` | 3 | HMM states: integer ≥ 2, or `auto` (BIC selection, 2–6) |
| `--min-train` | 252 | Warmup bars before walk-forward trading begins |
| `--saliency-threshold` | 0.5 | Features with saliency < threshold are pruned |
| `--saliency-output` | (none) | CSV file path to save per-feature saliency weights |

#### Strengths

- **Automatic feature selection** — no manual feature engineering needed.
- **Overfitting protection** — irrelevant features are silenced, not just
  regularised.
- **Best risk-adjusted performance** — in live testing on established equities
  with long histories, fshmm consistently produces the highest Sharpe and
  lowest drawdown.
- **Interpretable** — the saliency weights tell you *which* features matter
  for each asset.

#### Weaknesses

- **Slowest engine** — the saliency EM is computationally heavier than standard
  EM, and walk-forward re-fitting multiplies the cost.
- **Convergence noise** — the saliency EM can oscillate, producing "Model is
  not converging" warnings in stderr. These are harmless but may concern
  users.
- **Short-history risk** — with < 1000 bars, saliency may select too few
  features or pick calendar features by accident.

#### When to use

- Long histories (2000+ bars) where saliency has enough data to learn.
- When you want the best risk-adjusted output and can tolerate slower runs.
- For feature discovery — use `--saliency-output` to see which features
  the model considers important for each asset.

---

## Common Parameters

These parameters apply to all engines.

| Parameter | Default | Description |
|---|---|---|
| `--min-train` | 252 | Minimum bars before walk-forward trading starts. Lower values trade earlier but with less stable models. |
| `--dwell-bars` | 0 | **Dwell-time filter.** Require N consecutive same-regime bars before switching position. 0 = disabled. |
| `--hysteresis` | 0.0 | **Hysteresis filter.** Require posterior probability margin > D to switch regime. 0.0 = disabled. HMM engines only. |
| `--n-states` | 3 | Number of HMM states. Integer ≥ 2, or `auto` for BIC-based selection (2–6). HMM engines only. |
| `--duration-forecast` | off | Enable regime duration forecasting via survival analysis. See [Duration Forecasting](#duration-forecasting). |
| `--duration-model` | `weibull` | Survival model: `weibull` or `cox` (requires `lifelines`). |
| `--json` | off | Output structured JSON to stdout. Suppresses terminal pretty-print. |

### Output Format

| Flag | Behaviour |
|---|---|
| `--json` | JSON object to stdout. Suitable for piping into `jq`, scripts, etc. |
| (default) | Human-readable terminal output to stderr. |

---

## Walk-Forward Filters

The `--dwell-bars` and `--hysteresis` parameters control post-processing
filters applied to the raw regime labels during walk-forward backtesting.
They do **not** affect the regime labels themselves — only the trading
positions derived from them.

### Dwell-Time Filter (`--dwell-bars N`)

Requires N consecutive bars with the same regime label before switching
trading position. This eliminates brief whipsaw flips.

```bash
# Require 3 consecutive bull bars before going long
python -m hmm_futures_analysis.cli --ticker SPY --engine hmm --dwell-bars 3 --json
```

**Effect:** Reduces trade count, delays entry, eliminates transient regime blips.

### Hysteresis Filter (`--hysteresis D`)

Requires the posterior probability of the "winning" regime to exceed the
runner-up by at least D before switching. Only works with HMM engines
(which produce posterior probabilities).

```bash
# Require 20% probability margin to switch
python -m hmm_futures_analysis.cli --ticker SPY --engine hmm --hysteresis 0.2 --json
```

**Effect:** Reduces uncertainty-driven switches. The model must be "confident"
in the new regime before acting.

> Both filters can be combined: `--dwell-bars 3 --hysteresis 0.1`.

---

## Duration Forecasting

The `--duration-forecast` flag adds a survival analysis post-processor that
estimates **how long the current regime is likely to last**.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine messina --duration-forecast --json
```

### Weibull Model (default)

Fits a Weibull distribution to the historical spell lengths of the current
regime. Returns:

| Field | Meaning |
|---|---|
| `current_regime` | Which regime you're in (bear/sideways/bull) |
| `days_in_regime` | Bars spent in the current regime so far |
| `expected_remaining_days` | Expected bars remaining (conditional on survival so far) |
| `hazard_rate` | Instantaneous risk of regime ending at this moment |
| `survival_50pct` | Median total spell length for this regime |
| `weibull_shape` | Shape parameter k — k > 1 means "aging" (regime becomes more fragile over time), k < 1 means "infant mortality" (early exit risk) |
| `weibull_scale` | Scale parameter λ — characteristic lifetime |

### Cox PH Model (`--duration-model cox`)

Extended model using **realised volatility** and **spell return** as
covariates. Requires the `lifelines` package:

```bash
uv pip install lifelines
python -m hmm_futures_analysis.cli --ticker SPY --engine fshmm --duration-forecast --duration-model cox --json
```

Returns additional fields:

| Field | Meaning |
|---|---|
| `cox_coefficients` | Covariate effect sizes |
| `concordance_index` | Model fit quality (0.5 = random, 1.0 = perfect) |
| `baseline_hazard_at_t` | Baseline hazard at current duration |
| `cox_expected_remaining_days` | Covariate-adjusted expected remaining days |

### Requirements

The duration forecast needs at least **3 completed spells** for the current
regime. On short histories with sticky regimes, there may not be enough
spells to fit, and the output will contain `null` values.

---

## Output Reference

### JSON Structure

```json
{
  "source": "SPY",
  "engine": "fshmm",
  "dates": { "start": "2016-05-31", "end": "2026-05-29" },
  "current_regime": { "name": "bull", "index": 2 },
  "signal": 0.82,
  "next_state_probabilities": { "bear": 0.02, "sideways": 0.16, "bull": 0.82 },
  "transition_matrix": [[0.85, 0.10, 0.05], [0.03, 0.92, 0.05], [0.01, 0.07, 0.92]],
  "stationary_distribution": { "bear": 0.10, "sideways": 0.45, "bull": 0.45 },
  "persistence_diagonal": { "bear": 0.85, "sideways": 0.92, "bull": 0.92 },
  "regime_counts": { "bear": 200, "sideways": 1200, "bull": 1100 },
  "walk_forward": {
    "sharpe": 0.65,
    "max_drawdown": -0.24,
    "n_trades": 20,
    "win_rate": 0.55,
    "profit_factor": 4.88,
    "total_return": 2.07
  },
  "forecast": {
    "1_step":  { "bear": 0.02, "sideways": 0.16, "bull": 0.82 },
    "5_step":  { "bear": 0.07, "sideways": 0.28, "bull": 0.65 },
    "20_step": { "bear": 0.15, "sideways": 0.42, "bull": 0.43 }
  },
  "engine_info": {
    "method": "fshmm",
    "features": "generic",
    "n_states": 3,
    "warmup_bars": 252,
    "caveat": "HMM states sorted by mean return; labels may swap on re-fit",
    "feature_saliency": [0.32, 0.85, ...],
    "selected_features": ["log_ret", "rsi_14", ...]
  },
  "verdict": {
    "verdict": "bullish",
    "confidence": 0.82
  },
  "duration_forecast": {
    "current_regime": "bull",
    "days_in_regime": 45,
    "expected_remaining_days": 32.5,
    "hazard_rate": 0.0154,
    "survival_50pct": 60.0,
    "weibull_shape": 1.25,
    "weibull_scale": 80.0
  },
  "timing": {
    "total_s": 12.4,
    "phases": { "feature_engineering": 1.2, "regime_classification": 8.1, "walk_forward": 2.3, "forecast": 0.1 },
    "walk_forward_classify_stats": { "min": 0.001, "median": 0.012, "p99": 0.045, "n_calls": 2000 }
  },
  "framework": "hmm_test v0.2.0",
  "disclaimer": "Regime detection is probabilistic. Past transitions do not guarantee future regimes. Not financial advice."
}
```

### Key Fields Explained

| Field | Range | Interpretation |
|---|---|---|
| `signal` | −1 to +1 | `P(bull) − P(bear)` from next-state transition. > 0 = bullish lean. |
| `persistence_diagonal.*` | 0 to 1 | Self-transition probability. 0.95+ means very sticky regime. |
| `walk_forward.sharpe` | real | Annualised Sharpe ratio of walk-forward strategy. |
| `walk_forward.total_return` | real | Cumulative return (1.0 = +100%, −0.5 = −50%). |
| `walk_forward.profit_factor` | ≥ 0 | Gross profits / gross losses. > 1 = profitable. 0 = no winning trades. |
| `verdict.verdict` | string | One of `"bullish"`, `"bearish"`, `"neutral"`, `"transition_bull"`, `"transition_bear"`. Always present. |
| `verdict.confidence` | 0 to 1 | `abs(signal)`. Higher = stronger conviction. |
| `timing.total_s` | float | Total wall-clock time in seconds. Always present. |
| `timing.phases` | dict | Per-phase timing: `feature_engineering`, `regime_classification`, `walk_forward`, `forecast`. |

---

## Engine Selection Guide

### By Data Length

| Bars Available | Recommended Engine | Why |
|---|---|---|
| < 500 | `threshold` | HMM fitting unreliable on short data |
| 500 – 1000 | `threshold` or `messina` | 19 features is the maximum safe feature count |
| 1000 – 2000 | `messina` or `fshmm` | Saliency starts to become reliable |
| 2000+ | `fshmm` | Full feature set with automatic selection shines |

### By Asset Type

| Asset | Recommended Engine | Reasoning |
|---|---|---|
| Large-cap equities | `fshmm` | Long history, clean data — saliency selects the right features |
| Small-cap / volatile | `robust_hmm` | Outlier resistance protects against gap risk |
| Crypto | `robust_hmm` or `messina` | High volatility and gaps; robust fitting helps |
| Futures / indices | `messina` | Messina features designed for trend-following markets |
| Unknown / new asset | `threshold` | Safe baseline, no OHLCV requirement |

### By Use Case

| Goal | Recommended Engine | Parameters |
|---|---|---|
| Quick regime check | `threshold` | defaults |
| Stable signals, low turnover | `messina` | defaults |
| Best risk-adjusted returns | `fshmm` | `--saliency-threshold 0.5` |
| Feature discovery | `fshmm` | `--saliency-output features.csv` |
| Outlier-prone data | `robust_hmm` | `--robust-method mcd` |
| Reduce whipsaw | any HMM | `--dwell-bars 3 --hysteresis 0.1` |
| How long will this regime last? | any | add `--duration-forecast` |
| Custom state count | any HMM | `--n-states auto` or `--n-states 4` |

---

---

## Eval Mode

Run multiple engines across multiple tickers for batch comparison.

### `--eval-csv` — Evaluate from CSV directory

```bash
python -m hmm_futures_analysis.cli --eval-csv test_data/eval-results/ --json
```

Reads all `.csv` files from the directory. The filename stem (without extension)
becomes the ticker label. Each CSV is run through all engines (or a subset with
`--eval-engines`).

### `--eval-tickers` — Evaluate from ticker symbols

```bash
python -m hmm_futures_analysis.cli --eval-tickers CRM,0700.HK,SPY --json
```

Fetches each ticker via yfinance and runs all engines. Use `--eval-cache-dir`
to save the downloaded data as CSVs for reproducibility:

```bash
python -m hmm_futures_analysis.cli --eval-tickers CRM,SPY \
  --eval-cache-dir ./csv-cache/ --json
```

### `--eval-engines` — Filter engines

```bash
# Only run threshold and messina
python -m hmm_futures_analysis.cli --eval-csv data/ --eval-engines threshold,messina --json
```

Default: all five engines (`threshold`, `messina`, `hmm`, `robust_hmm`, `fshmm`).

### Output

| Flag | Output |
|---|---|
| (default) | Markdown comparison table to **stderr** |
| `--json` | JSON array to **stdout** |

#### Table columns

| Column | Description |
|---|---|
| ticker | Ticker symbol or CSV filename stem |
| engine | Engine name |
| regime | Current regime (bear/sideways/bull) |
| signal | Bull−bear signal (−1 to +1) |
| sharpe | Walk-forward annualised Sharpe ratio |
| max_dd | Maximum drawdown |
| trades | Number of walk-forward trades |
| win_rate | Fraction of profitable trades |
| pf | Profit factor (gross wins / gross losses) |
| total_ret | Cumulative return |
| wall_s | Wall-clock time in seconds |

### Flags summary

| Flag | Description |
|---|---|
| `--eval-tickers LIST` | Comma-separated ticker symbols |
| `--eval-csv DIR` | Directory of CSV files |
| `--eval-engines LIST` | Comma-separated engine names (default: all) |
| `--eval-cache-dir DIR` | Save yfinance data to CSV (used with `--eval-tickers`) |

> `--eval-tickers` and `--eval-csv` are mutually exclusive with each other and with
> `--ticker`/`--csv`.

---

## Common Issues

### "ta library not available" warnings

The `ta` Python package provides the technical indicators (RSI, MACD,
Bollinger bands, ADX, etc.) used by `hmm`, `robust_hmm`, and `fshmm`. If
it's not installed, these engines run with a degraded feature set.

**Fix:** Ensure the virtual environment is activated, then:

```bash
uv sync
python -c "import ta; print('OK')"  # verify
```

### "Model is not converging" warnings

The fshmm engine prints these during walk-forward EM fitting. They indicate
that the log-likelihood oscillated slightly between iterations. The model
recovers and produces valid output. These are informational, not errors.

### All regimes are "sideways" or all "bear"

This typically happens with:
- **Too few bars** — the model doesn't have enough data to distinguish states.
- **Too many features** — overfitting collapses the model to 1–2 effective states.

**Fix:** Use `threshold` or `messina` instead, or increase `--min-train`.

### Walk-forward Sharpe is deeply negative

The HMM detected regimes but the trading strategy was consistently wrong.
Common with `hmm` on secular-trend stocks where the model labels most of a
decade-long bull market as "bear."

**Fix:** Try `fshmm` (automatic feature selection often corrects this) or
`messina` (curated features resist this failure mode).

---

## Examples

```bash
# Quick check on Apple
python -m hmm_futures_analysis.cli --ticker AAPL --json

# Best-quality analysis on S&P 500
python -m hmm_futures_analysis.cli --ticker "^GSPC" --engine fshmm \
  --duration-forecast --dwell-bars 3 --json

# Bitcoin with robust fitting
python -m hmm_futures_analysis.cli --ticker BTC-USD --engine robust_hmm \
  --robust-method mcd --json

# Custom CSV with auto state count
python -m hmm_futures_analysis.cli --csv my_data.csv --engine hmm \
  --n-states auto --json

# Feature saliency discovery
python -m hmm_futures_analysis.cli --ticker MS --engine fshmm \
  --saliency-output ms_saliency.csv --json

# Duration forecast with Cox PH model
python -m hmm_futures_analysis.cli --ticker KO --engine messina \
  --duration-forecast --duration-model cox --json

# Terminal (non-JSON) output for human reading
python -m hmm_futures_analysis.cli --ticker MS --engine fshmm
```
