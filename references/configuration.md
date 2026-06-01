# Configuration Reference

## CLI Parameters

### `--engine` (default: `threshold`)

Analysis engine for regime detection. Selects the algorithm and feature set.

| Value | Description |
|-------|-------------|
| `threshold` | Fast, close-only rolling return threshold. No HMM fitting needed. |
| `messina` | HMM + 19 curated Messina features (OHLCV required). Wilder's smoothing. |
| `hmm` | HMM + ~50 generic technical indicators (OHLCV required). Full `ta` library. |
| `robust_hmm` | Same as `hmm` but with outlier-resistant emission estimation (Huber/MCD). |
| `fshmm` | Same as `hmm` but with automatic feature saliency pruning (Adams et al. 2016). |

**Engine selection guidance**:
- **< 500 bars**: Use `threshold` — HMM fitting unreliable on short histories.
- **500–1000 bars**: `threshold` or `messina` (19 features is the safe max).
- **1000–2000 bars**: `messina` or `fshmm` — saliency starts to become reliable.
- **2000+ bars**: `fshmm` — full feature set with automatic selection.
- **Crypto / volatile assets**: `robust_hmm` or `messina` for outlier resistance.
- **Quick baseline**: `threshold` — no OHLCV requirement.

### `--window` (default: 20)

Rolling window size for regime classification. Number of bars over which the return sum is computed.

| Value | Effect |
|-------|--------|
| 5–10 | Very responsive. Frequent regime switches. High noise. |
| 20 | Balanced. Default for daily data. |
| 50–60 | Slow. Captures only persistent trends. Fewer trades. |
| 126 (6 months) | Long-term regime. Ignores short-term fluctuations. |

**Guidelines by frequency:**
- **Daily**: 10–20 (2–4 weeks of data)
- **Weekly**: 8–13 (2–3 months of data)
- **Hourly**: 20–50 (1–2 days of data)
- **Minute**: 50–100 (1–2 hours of data)

### `--threshold` (default: 0.05)

Return threshold for bull/bear classification. The absolute value of the rolling return sum must exceed this to classify a bar as bull or bear.

| Value | Effect |
|-------|--------|
| 0.01–0.02 | Sensitive. Captures small moves. High turnover. |
| 0.05 | Moderate. Default for daily equity/futures. |
| 0.10–0.15 | Conservative. Only large moves trigger regime changes. |

**Guidelines by asset class:**
- **Equities** (SPY, QQQ): 0.03–0.05
- **Futures** (ES, NQ): 0.02–0.05 (leverage amplifies returns)
- **Forex**: 0.01–0.03 (lower daily volatility)
- **Crypto** (BTC, ETH): 0.05–0.10 (higher volatility)

### `--min-train` (default: 252)

Minimum number of bars before walk-forward trading starts. Represents the "warm-up" period.

| Value | Context |
|-------|---------|
| 252 | 1 year of daily data. Standard default. |
| 126 | 6 months. Higher turnover, more trades. |
| 504 | 2 years. Conservative, needs more data. |
| 50 | 50 bars. For short histories or high-frequency data. |

The walk-forward backtest returns `null` for Sharpe and 0 trades if `len(prices) < min_train + 1`. Increase this for more stable transition matrix estimates.

### `--n-states` (default: 3)

Number of HMM hidden states for HMM engines (messina, hmm, robust_hmm, fshmm).

| Value | Use case |
|-------|----------|
| 2 | Binary up/down. Simplest. Less granular. |
| 3 | Bear/sideways/bull. Standard for regime detection. |
| 4 | Strong bear / weak bear / weak bull / strong bull. |
| 5+ | Sub-regime analysis. Requires >1000 bars. Hard to interpret. |

**Recommendation**: Start with 3. Only increase if AIC/BIC shows significant improvement and the additional states have clear economic interpretation.

### `--dwell-bars` (default: 0)

Dwell-time filter for walk-forward backtesting. Requires N consecutive same-regime bars before switching position. Eliminates brief whipsaw flips.

| Value | Effect |
|-------|--------|
| 0 | Disabled (default). No dwell filter applied. |
| 1–3 | Light filtering. Eliminates single-bar flips. |
| 3–5 | Moderate. Good for daily data on volatile assets. |
| 5+ | Heavy. Only persistent regime changes trigger switches. |

**Recommendation**: Start with 3 for daily data. Combine with `--hysteresis` for strongest whipsaw protection.

### `--hysteresis` (default: 0.0)

Hysteresis filter for walk-forward backtesting. Requires posterior probability margin > D to switch regime. Only applies to HMM engines (which produce posterior probabilities).

| Value | Effect |
|-------|--------|
| 0.0 | Disabled (default). |
| 0.05–0.10 | Light. Only low-confidence switches are suppressed. |
| 0.10–0.20 | Moderate. Model must be fairly confident to switch. |
| 0.20+ | Aggressive. Only very confident regime changes pass. |

**Recommendation**: Start with 0.10. Combine with `--dwell-bars 3` for best results.

### `--robust-method` (default: `huber`)

Robust estimation method for the `robust_hmm` engine. Controls how outliers are handled when computing initial emission parameters.

| Value | Description |
|-------|-------------|
| `huber` | Iteratively Reweighted Least Squares with Huber loss. Faster. Default. |
| `mcd` | Minimum Covariance Determinant (scikit-learn). More aggressive outlier rejection. |

**Recommendation**: Start with `huber`. Use `mcd` for assets with extreme outliers (flash crashes, earnings gaps).

### `--saliency-threshold` (default: 0.5)

Feature saliency pruning threshold for the `fshmm` engine. Features with learned saliency weight below this threshold are automatically masked as irrelevant during EM training.

| Value | Effect |
|-------|--------|
| 0.3 | Lower threshold. More features retained. Useful for feature discovery. |
| 0.5 | Default balance. Prunes only clearly irrelevant features. |
| 0.7–0.9 | Aggressive pruning. Only high-saliency features survive. |

### `--saliency-output` (default: none)

File path to save fshmm per-feature saliency weights as CSV. The CSV contains columns: `feature_index`, `saliency_weight`, `selected`.

```bash
python -m hmm_futures_analysis.cli --ticker SPY --engine fshmm --saliency-output saliency.csv
```

### `--duration-forecast` (flag, default: off)

Enable regime duration forecasting via survival analysis. When enabled, the output includes expected remaining time in the current regime, hazard rate, and Weibull distribution parameters.

See [USAGE.md Duration Forecasting](../USAGE.md#duration-forecasting) for full details.

### `--duration-model` (default: `weibull`)

Survival model for duration forecasting.

| Value | Description |
|-------|-------------|
| `weibull` | Standard Weibull distribution. Default. Works on any history with 3+ spells. |
| `cox` | Cox Proportional Hazards model with realised volatility and spell return as covariates. Requires `lifelines`. |

## HMM Hyperparameters (programmatic access)

These are set when constructing engine config dataclasses directly (see `engine_protocol.py`), not via CLI:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `covariance_type` | `"full"` | Covariance matrix structure. `"full"` = each state has own full covariance. `"diag"` = diagonal only (fewer params, faster). `"tied"` = all states share one covariance. `"spherical"` = single variance per state. |
| `n_iter` | 100 | Maximum EM iterations. Increase for difficult convergence cases. |
| `random_state` | 42 | Seed for reproducibility. |
| `tol` | 1e-3 | Convergence threshold. Smaller = tighter convergence but more iterations. |
| `model_type` | `"gaussian"` | `"gaussian"` for standard HMM, `"gmm"` for Gaussian Mixture Model HMM. |

### Covariance Type Trade-offs

| Type | Parameters per state | When to use |
|------|---------------------|-------------|
| `full` | `n_features²` | Best fit, most flexible. Default for <10 features. |
| `diag` | `n_features` | Good for many features (>10). Faster convergence. |
| `tied` | shared | When regimes have similar volatility structure. |
| `spherical` | 1 | Simplest. For 1–2 feature models. |

### Config Dataclass Parameters

These fields are set on the config dataclass constructors, not passed to hmmlearn:

| Parameter | Engines | Default | Description |
|-----------|---------|---------|-------------|
| `pca_variance` | HMM engines | `None` | If set (e.g. `0.95`), features are whitened via PCA retaining this fraction of variance. Not exposed via CLI; set programmatically. |
| `saliency_threshold` | `fshmm` | `0.5` | Feature saliency pruning threshold. CLI equivalent: `--saliency-threshold`. |
| `robust_method` | `robust_hmm` | `"huber"` | Robust estimator method. CLI equivalent: `--robust-method`. |

## Parameter Tuning

### Grid Search Pattern

#### Threshold Engine Tuning

```python
from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine
from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

best_sharpe = -float("inf")
best_params = {}

for window in [10, 20, 30]:
    for threshold in [0.03, 0.05, 0.07]:
        engine = ThresholdEngine(window=window, threshold=threshold)
        result = walk_forward_backtest(prices, engine=engine)
        if result["sharpe"] and result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_params = {"window": window, "threshold": threshold}
```

#### HMM Engine Tuning

```python
from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine
from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

best_sharpe = -float("inf")
best_params = {}

for n_states in [2, 3, 4]:
    for dwell_bars in [0, 3, 5]:
        for hysteresis in [0.0, 0.1, 0.2]:
            engine = HMMGenericEngine(n_states=n_states)
            result = walk_forward_backtest(
                prices,
                engine=engine,
                dwell_bars=dwell_bars,
                hysteresis_delta=hysteresis,
            )
            if result["sharpe"] and result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                best_params = {
                    "n_states": n_states,
                    "dwell_bars": dwell_bars,
                    "hysteresis": hysteresis,
                }
```

**Note**: HMM engines require precomputed features from OHLCV data. For proper walk-forward tuning, pass precomputed features and an OHLCV DataFrame through the pipeline. The grid search above illustrates the pattern; see [USAGE.md HMM Parameters](../USAGE.md#2-hmm--50-generic-features) for full pipeline examples.

**Warning**: Grid searching on the same data used for evaluation introduces selection bias. Use a separate validation period or walk-forward cross-validation.

### Recommended Defaults by Asset Class

| Asset Class | engine | window | threshold | min_train |
|------------|--------|--------|-----------|-----------|
| US Large Cap Equities | `fshmm` | 20 | 0.05 | 252 |
| US Small Cap Equities | `robust_hmm` | 20 | 0.06 | 252 |
| Index Futures (ES) | `messina` | 15 | 0.04 | 252 |
| Commodity Futures | `messina` | 20 | 0.05 | 252 |
| Major Forex (EURUSD) | `threshold` | 20 | 0.02 | 252 |
| Crypto (BTC) | `robust_hmm` | 10 | 0.07 | 126 |
| Intraday (1H bars) | `messina` | 30 | 0.01 | 200 |
