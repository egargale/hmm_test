# Configuration Reference

## CLI Parameters

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

Number of HMM hidden states. Only relevant when `--hmm` (default).

| Value | Use case |
|-------|----------|
| 2 | Binary up/down. Simplest. Less granular. |
| 3 | Bear/sideways/bull. Standard for regime detection. |
| 4 | Strong bear / weak bear / weak bull / strong bull. |
| 5+ | Sub-regime analysis. Requires >1000 bars. Hard to interpret. |

**Recommendation**: Start with 3. Only increase if AIC/BIC shows significant improvement and the additional states have clear economic interpretation.

### `--hmm` / `--no-hmm`

| Flag | Effect |
|------|--------|
| (default) | Run both threshold classification and HMM analysis. HMM failure is non-fatal. |
| `--hmm` | Explicitly enable (same as default). |
| `--no-hmm` | Skip HMM entirely. Faster, no hmmlearn dependency required. |

## HMM Hyperparameters (programmatic access)

These are set when calling `run_hmm_regime()` directly, not via CLI:

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

## Parameter Tuning

### Grid Search Pattern

```python
from regime.markov_chain import classify_regimes
from regime.walk_forward import walk_forward_backtest

best_sharpe = -float("inf")
best_params = {}

for window in [10, 20, 30]:
    for threshold in [0.03, 0.05, 0.07]:
        result = walk_forward_backtest(prices, window=window, threshold=threshold)
        if result["sharpe"] and result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_params = {"window": window, "threshold": threshold}
```

**Warning**: Grid searching on the same data used for evaluation introduces selection bias. Use a separate validation period or walk-forward cross-validation.

### Recommended Defaults by Asset Class

| Asset Class | window | threshold | min_train |
|------------|--------|-----------|-----------|
| US Large Cap Equities | 20 | 0.05 | 252 |
| US Small Cap Equities | 20 | 0.06 | 252 |
| Index Futures (ES) | 15 | 0.04 | 252 |
| Commodity Futures | 20 | 0.05 | 252 |
| Major Forex (EURUSD) | 20 | 0.02 | 252 |
| Crypto (BTC) | 10 | 0.07 | 126 |
| Intraday (1H bars) | 30 | 0.01 | 200 |
