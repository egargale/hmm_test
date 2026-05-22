# Feature Gap Scout Report: Messina vs Generic HMM Modes

## Overview

The `--messina` CLI flag (defined in `scripts/cli.py`, line 319) selects between two entirely different feature engineering pipelines before feeding the same HMM model. The downstream model code (`GaussianHMMModel` / `GMMHMMModel`) is completely agnostic to which features it receives — it just receives a 2D float64 array after NaN-dropping and z-score standardization.

---

## 1. Generic Feature Set (the 44-feature set)

**Entry point:** `scripts/data_processing/feature_engineering.py`, function `add_features()` (line 28)

**Signature:**
```python
def add_features(
    df: pd.DataFrame,
    indicator_config: Optional[Dict[str, Dict[str, Any]]] = None,
    downcast_floats: bool = True,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
```

When called without `indicator_config`, it uses the default config from `scripts/data_processing/technical_indicators.py`, function `get_default_indicator_config()` (line 40). This produces features in 13 categories:

| Category | Features (exact column names) |
|---|---|
| Basic returns | `log_ret`, `simple_ret` |
| Moving averages | `sma_20`, `ema_20`, `sma_ratio_50_200` |
| Volatility | `atr_14`, `bb_upper_20`, `bb_middle_20`, `bb_lower_20`, `bb_width_20`, `bb_position_20` |
| Momentum | `rsi_14`, `roc_10`, `stoch_k_14`, `stoch_d_3` |
| Volume | `volume_sma_20`, `volume_ratio_20`, `obv`, `vwap` |
| Trend | `adx_14`, `adx_plus_14`, `adx_minus_14` |
| Price patterns | `price_position_5`, `price_position_10`, `price_position_20`, `hl_ratio_5`, `hl_ratio_10`, `hl_ratio_20` |
| Enhanced momentum | `williams_r_14`, `cci_20`, `mfi_14` |
| Enhanced volatility | `chaikin_vol_10_10`, `hv_20`, `keltner_upper/lower/middle_20_10`, `donchian_upper/lower/middle_20` |
| Enhanced trend | `tma_20`, `aroon_up/down/oscillator_25`, `di_plus/minus_14`, `adx_14` |
| Enhanced volume | `adl`, `vpt`, `eom_20`, `volume_roc_20` |
| Time features | `day_of_week`, `day_of_month`, `month`, `quarter`, `is_month_end`, `is_month_start`, `is_quarter_end`, `day_of_week_sin/cos`, `month_sin/cos`, `hour`, `minute`, `session`, `hour_sin/cos`, `is_weekend` |
| Custom | (empty by default) |

**Total: ~40–50 features** depending on data length and config.

**How it's consumed in non-Messina mode** (`scripts/regime/hmm_adapter.py`, lines 46–53):
```python
df = add_features(prices, min_periods=10)
df = df.dropna(axis=1, how="all")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
ohlcv = {"open", "high", "low", "close", "volume"}
numeric_cols = [c for c in numeric_cols if c not in ohlcv]
```
All numeric non-OHLCV columns pass through to the HMM. No feature selection is applied.

---

## 2. Messina Feature Set

**Entry point:** `scripts/data_processing/messina_features.py`, function `add_messina_features()` (line 103)

**Signature:**
```python
def add_messina_features(
    df: pd.DataFrame,
    vstop_multiplier: float = 2.0,
) -> pd.DataFrame:
```

**Produced columns (exactly 12):**

| Column | Calculation | Notes |
|---|---|---|
| `log_ret` | `ln(close / close.shift(1))` | Same as generic |
| `sma_200` | 200-bar SMA | Long-term trend |
| `sma_13` | 13-bar SMA | Medium-term trend |
| `atr_20` | Wilder's smoothed ATR (20) | Volatility |
| `adx_14` | Wilder's smoothed ADX (14) | Trend strength |
| `di_plus_14` | Wilder's smoothed +DI (14) | Upward directional movement |
| `di_minus_14` | Wilder's smoothed -DI (14) | Downward directional movement |
| `adx_slope` | `adx_14.diff(3)` | ADX momentum |
| `vstop` | Volatility stop (SMA13 ± 2×ATR20) | Trailing stop level |
| `vstop_trend` | 1=uptrend, -1=downtrend | Binary trend signal |
| `price_sma200_ratio` | `close / sma_200` | Distance from long-term MA |
| `price_vstop_ratio` | `close / vstop` | Distance from trailing stop |

**Key differences from generic:**
- Uses **Wilder's smoothing** (modified EMA) for ATR, ADX, and DI — not the standard SMA-based `ta` library calculations used in generic mode.
- Includes a custom **Volatility Stop** (`_calc_vstop`) which is a trailing stop mechanism — no equivalent in generic mode.
- `adx_slope` gives directional change of trend strength — no equivalent in generic mode.
- **No momentum oscillators** (no RSI, no CCI, no stochastic, no MACD).
- **No volume indicators** (no OBV, no VWAP, no volume ratio).
- **No time-based features** (no day-of-week, no session, no cyclical encoding).

**How it's consumed** (`scripts/regime/hmm_adapter.py`, lines 38–45):
```python
df = add_messina_features(prices)
messina_cols = [
    "log_ret", "sma_200", "sma_13", "atr_20",
    "adx_14", "di_plus_14", "di_minus_14", "adx_slope",
    "vstop", "vstop_trend", "price_sma200_ratio", "price_vstop_ratio",
]
numeric_cols = [c for c in messina_cols if c in df.columns]
```
Exactly 12 columns are selected by name.

---

## 3. HMM Model — How It's Trained

Both `GaussianHMMModel` (`scripts/hmm_models/gaussian_hmm.py`) and `GMMHMMModel` (`scripts/hmm_models/gmm_hmm.py`) follow the same pipeline:

### `_prepare_features()` (gaussian_hmm.py line 80, gmm_hmm.py line 89)
1. Drops rows with any NaN
2. Converts to `float64` numpy array
3. **Standardizes (z-scores)**: `(X - mean(X, axis=0)) / (std(X, axis=0) + 1e-8)`
4. Returns clean 2D array

### Training
- Uses `hmmlearn`'s `GaussianHMM` or `GMMHMM`
- Default 3 states, full covariance, 100 EM iterations
- Multiple random restarts (controlled by `HMMConfig.num_restarts` in `scripts/model_training/hmm_trainer.py`)
- Best model selected by log-likelihood

### Data shape expectation
- **Any N-column 2D array** is accepted — the model has zero awareness of feature semantics
- Feature dimension must match between training and inference (enforced at line 60 of `inference_engine.py`)

### State labeling (hmm_adapter.py, lines 57–64)
After training, states are sorted by `log_ret` mean (or `means_[:, 0]` if `log_ret` not found):
- Lowest mean → bear (state 0)
- Middle → sideways (state 1)
- Highest → bull (state 2)

This is a **critical fragility**: if `log_ret` is not the first column or isn't present, the fallback is `means_[:, 0]` which may have no semantic meaning.

---

## 4. Feature Mode Selection in the Factory

**File:** `scripts/hmm_models/factory.py`

The factory (`HMMModelFactory`) only selects the **model type** (gaussian vs gmm) — it has **nothing to do with feature selection**. The Messina vs Generic choice happens upstream in `scripts/regime/hmm_adapter.py` at line 38:

```python
if use_messina:
    df = add_messina_features(prices)
    messina_cols = [...]  # explicit whitelist
else:
    df = add_features(prices, min_periods=10)
    # all numeric non-OHLCV columns
```

The `--messina` / `--no-messina` flag is rendered in `scripts/cli.py` (lines 319–323) and passed to `build_json_output()` → `run_hmm_regime()`.

---

## 5. Tradable Signals from HMM

The HMM produces **states** (0, 1, 2), which are labeled bear/sideways/bull. The pipeline then:

1. Computes `stationary_distribution` from the transition matrix
2. Sorts states by `log_ret` mean to assign labels
3. Returns `regimes` (list of strings), `transition_matrix`, and `stationary_distribution` in the JSON output

The **actual tradable signal** is produced by `compute_signal()` in `scripts/regime/markov_chain.py` (line 67):
```python
def compute_signal(next_state_probs: np.ndarray) -> float:
    return float(next_state_probs[2] - next_state_probs[0])
```
This gives a value in [-1, 1]. The signal is used in `walk_forward_backtest()` (walk_forward.py line 65) for position sizing: `np.clip(signal, -1.0, 1.0)`.

**However:** the walk-forward backtest in `walk_forward.py` uses **threshold-based regimes** (rolling returns), ***not*** HMM states. The HMM is a separate analysis block in the JSON output. The `pipeline.py` calls `walk_forward_backtest()` independently of the HMM result. HMM results are purely informational/analytical in the current architecture — they don't feed the backtest P&L.

---

## 6. Fair Comparison: Messina vs Generic HMM Backtests

### Feasibility: ✅ Possible but requires manual orchestration

**Why it's possible:**
- Both modes use the **same model code** (`GaussianHMMModel` or `GMMHMMModel`)
- Both go through the same `_prepare_features()` → standardize → fit → predict flow
- The CLI already supports both with `--messina` / no flag
- The output `feature_mode` field ("messina" vs "generic") in the HMM result dict (`hmm_adapter.py` line 103) makes comparison straightforward

**What makes comparison tricky:**

| Issue | Details |
|---|---|
| **Different feature dimensionality** | Generic ~44 features, Messina exactly 12. A 44-feature `full` covariance HMM has ~44×45/2 = 990 covariance parameters per state vs 12×13/2 = 78 for Messina. This is an enormous complexity difference that will affect convergence and state separation. |
| **Different calculations for same-named features** | ADX/DI in Messina uses **Wilder's smoothing** (exponential). ADX/DI in generic mode uses `ta.trend.ADXIndicator` which uses SMA-based smoothing. These produce different values, so even intersecting features like `adx_14`, `di_plus_14`, `di_minus_14` are not directly comparable. |
| **Messina has unique features** | `vstop`, `vstop_trend`, `adx_slope`, `price_vstop_ratio` have no counterpart in generic mode. These are specifically designed for stop-loss / trailing logic. |
| **Generic has many features Messina lacks** | All momentum oscillators (RSI, CCI, MFI, stochastic), volume indicators, time features, Bollinger Bands, Keltner/Donchian channels, etc. This introduces potential overfitting. |
| **HMM backtest signal not used in walk-forward** | The current pipeline's walk-forward backtest uses threshold-based regimes only. To get HMM-driven P&L, you'd need to either: (a) use `scripts/backtesting/strategy_engine.py` with HMM states, or (b) modify the pipeline to feed HMM states into the walk-forward loop. |
| **State label fragility** | Both modes rely on `log_ret` being present for state labeling. Generic mode always has `log_ret`. Messina has `log_ret` but the hmm_adapter fallback to `means_[:, 0]` is risky if column ordering changes. |

### Recommendation for fair comparison

To compare Messina vs Generic HMM backtests fairly:

1. **Run two separate pipeline calls**: one with `--messina`, one without, on the same data.
2. **Use the `strategy_engine.py` backtester** (not the walk-forward) for both, passing HMM states directly.
3. **Control for model complexity**: either reduce generic features to a matched subset, or use `diag` covariance for generic mode to reduce parameter count.
4. **Address the Wilder's vs SMA difference** if comparing individual feature behavior.
5. **Both modes can and do produce tradable signals** in the form of labeled states → `compute_signal()` → position.

---

## Files Retrieved Summary

| File | Lines | Role |
|---|---|---|
| `scripts/data_processing/feature_engineering.py` | 28–185 (core) | Generic 44-feature set producer |
| `scripts/data_processing/technical_indicators.py` | 40–80 | Default indicator config (controls which features are produced) |
| `scripts/data_processing/messina_features.py` | 103–185 | Messina 12-feature set producer |
| `scripts/regime/hmm_adapter.py` | 1–108 | **Bridge layer**: selects feature mode, trains HMM, labels states |
| `scripts/hmm_models/gaussian_hmm.py` | 1–280 | Gaussian HMM model (feature-agnostic) |
| `scripts/hmm_models/gmm_hmm.py` | 1–370 | GMM HMM model (feature-agnostic) |
| `scripts/hmm_models/factory.py` | 1–280 | Model type factory (model type only, not features) |
| `scripts/hmm_models/base.py` | 1–290 | Base class with fit/predict/save/load |
| `scripts/model_training/hmm_trainer.py` | 1–250 | Training loop with restarts, scaling, validation |
| `scripts/model_training/inference_engine.py` | 1–380 | State prediction, lagged inference, stability analysis |
| `scripts/regime/markov_chain.py` | 66–68 | `compute_signal()` — converts state probs to [-1,1] signal |
| `scripts/regime/walk_forward.py` | 1–100 | Walk-forward backtest (uses threshold, not HMM) |
| `scripts/regime/pipeline.py` | 1–200 | Top-level pipeline; calls HMM as separate analysis block |

## Start Here

**`scripts/regime/hmm_adapter.py`** — it's the single file where feature mode selection, HMM training, state labeling, and output formatting all converge. Everything else feeds into or out of it.
