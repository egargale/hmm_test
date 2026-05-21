# Feature Engineering for Regime Detection

## Overview

The skill works at two levels of feature complexity:

| Mode | Features Used | Engine |
|------|--------------|--------|
| **Threshold** | Rolling return sum only | `regime.markov_chain.classify_regimes()` |
| **HMM** | Multiple engineered features | `data_processing.feature_engineering.add_features()` |

## Threshold Mode Features

The threshold classifier uses a single derived feature: **rolling return sum** over the `--window` period.

```
rolling_ret[t] = sum(returns[t-window+1 : t+1])
```

Classification rule:
- `rolling_ret > +threshold` → Bull (state 2)
- `rolling_ret < -threshold` → Bear (state 0)
- otherwise → Sideways (state 1)

This is computationally cheap, deterministic, and requires no model fitting. Use it for quick regime checks or when data is limited.

## HMM Mode Features

When `--hmm` is enabled (default), the HMM pipeline calls `add_features()` which computes a suite of technical indicators. The feature set includes:

### Price-based Features
| Feature | Description | Why it matters |
|---------|-------------|----------------|
| **Log-returns** | `log(close[t] / close[t-1])` | Primary signal — directly measures price movement direction and magnitude. |
| **Returns (1, 5, 10 period)** | Percentage change over multiple horizons | Captures momentum at different time scales. Short-horizon for noise, long-horizon for trend. |
| **Moving averages (5, 10, 20 period)** | Simple rolling mean of close | Smooths noise. Price vs MA crossovers are classical trend-change signals. |
| **Bollinger Bands** | SMA ± 2σ | Measures deviation from mean. Wide bands = high volatility regime. |

### Volatility Features
| Feature | Description | Why it matters |
|---------|-------------|----------------|
| **Historical volatility (14 period)** | Rolling standard deviation of returns | High-vol regimes differ from low-vol regimes. Bear markets typically exhibit higher volatility. |
| **ATR (Average True Range)** | Moving average of true range | Normalises volatility to price level. Useful across assets with different price scales. |

### Momentum Features
| Feature | Description | Why it matters |
|---------|-------------|----------------|
| **ROC (Rate of Change)** | Percentage change over N periods | Momentum indicator. Persistent positive ROC → bull trend. |
| **RSI (Relative Strength Index)** | 14-period normalised strength | Overbought/oversold signals. RSI > 70 suggests overbought (potential bear reversal). |
| **MACD** | 12/26-period EMA crossover | Trend-following. MACD > signal line = bullish momentum. |
| **Stochastic oscillator** | %K / %D crossover | Short-term momentum. Complements longer-term indicators. |

### Volume Features
| Feature | Description | Why it matters |
|---------|-------------|----------------|
| **Volume SMA** | Rolling mean of volume | High volume confirms trend strength. Low volume during trends = weak conviction. |
| **Volume ratio** | Current volume / SMA volume | Spikes often precede reversals. |

## Feature Selection for HMM

Not all features are used in state inference. The `add_features()` function returns a DataFrame with all computed features, but only **numeric non-OHLCV columns** are passed to the HMM. The model automatically handles feature covariance through the emission distribution.

For large datasets or slow fitting, consider reducing features by setting `indicators` in the ProcessingConfig or using the `feature_selection` module.

## Scaling

All features fed to the HMM are standardised (zero mean, unit variance) using `sklearn.preprocessing.StandardScaler`. This is critical because:

1. Features with larger numeric ranges (e.g. price 10,000 vs returns 0.001) would dominate the covariance matrix.
2. The HMM's Gaussian emission assumption is better satisfied with standardised inputs.
3. The scaler is fitted on training data only and applied consistently.

## Data Quality

Before feature engineering, the pipeline validates:
- No NaN or infinite values in input OHLCV data
- Consistent datetime ordering (sorted, no duplicates)
- Required columns present (open, high, low, close, volume)

The `data_processing.data_validation.validate_data()` function handles missing values (forward-fill by default) and outlier detection (IQR method by default).

## Custom Feature Engineering

To add custom features, extend `data_processing/feature_engineering.py` or pass a pre-engineered DataFrame directly to `run_hmm_regime()`:

```python
from regime.hmm_adapter import run_hmm_regime

# Your custom feature DataFrame with numeric columns
custom_features = compute_my_features(prices_df)
result = run_hmm_regime(custom_features, n_states=3)
```

The HMM will treat all numeric columns as emission features. Ensure columns are named descriptively for debugging.

## Memory Considerations

Feature engineering on large datasets can be memory-intensive. The `csv_parser.process_csv()` function supports chunked processing for files >100MB, with automatic float32 downcasting. For HMM training, consider downsampling (e.g. daily instead of hourly) if memory is constrained — the regime structure is typically preserved at coarser resolutions.
