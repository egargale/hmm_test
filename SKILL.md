---
name: hmm-regime-detection
description: >
  Detect the current market regime (Bull/Bear/Sideways) for any asset using
  Hidden Markov Models and Markov chain analysis. Use when the user wants
  regime detection, Markov transition analysis, walk-forward backtesting,
  or regime-based risk gating — on a ticker (via yfinance) or on CSV price data.
  Produces a structured JSON output compatible with the agentskills.io regime contract.
---

# HMM Regime Detection Skill

Detect whether an asset is in a **Bull** (uptrend), **Bear** (downtrend), or **Sideways** (range-bound) regime using threshold-based classification with optional Hidden Markov Model (HMM) analysis. Includes no-lookahead walk-forward backtesting so you can evaluate how regime-based trading signals would have performed historically.

## Quick Invocation

```bash
# CSV file (relative or absolute path)
python scripts/cli.py --csv BTC.csv --json

# yfinance ticker
python scripts/cli.py --ticker ES=F --json

# Threshold-only (skip HMM)
python scripts/cli.py --csv BTC.csv --json --no-hmm

# Custom parameters
python scripts/cli.py --ticker SPY --json --window 10 --threshold 0.03 --min-train 126
```

The `--json` flag sends one JSON object to stdout and nothing else. Without `--json`, a pretty-printed table goes to stderr.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv` | path | — | Path to CSV file with price data (relative or absolute). |
| `--ticker` | str | — | yfinance ticker (e.g. `ES=F`, `SPY`, `BTC-USD`). |
| `--json` | flag | off | Output JSON to stdout. On error: `{"error": "..."}` + exit 1. |
| `--window` | int | 20 | Rolling window for regime classification. |
| `--threshold` | float | 0.05 | Rolling return threshold for bull/bear classification. |
| `--min-train` | int | 252 | Minimum bars before walk-forward trading starts. |
| `--hmm` | flag | on | Enable HMM analysis (default). |
| `--no-hmm` | flag | — | Disable HMM (threshold only). |
| `--n-states` | int | 3 | Number of HMM states. |

## State Interpretation

| State | Index | Trading Action |
|-------|-------|----------------|
| **Bear** | 0 | Short (or exit longs) |
| **Sideways** | 1 | Flat / no position |
| **Bull** | 2 | Long |

The threshold method uses rolling return to classify: above `+threshold` = Bull, below `-threshold` = Bear, otherwise Sideways.

## JSON Output Contract

The JSON output is compatible with the agentskills.io regime contract. All agents consuming this skill should expect the following structure:

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | Ticker symbol or CSV filename. |
| `rows` | int | Number of price bars. |
| `date_start` | str | First date in the series (YYYY-MM-DD). |
| `date_end` | str | Last date in the series (YYYY-MM-DD). |
| `params` | object | Parameters used: `window`, `threshold`, `method`. |
| `states` | array | Metadata: `[{"name": "bear", "index": 0}, ...]`. |
| `current_regime` | object | `{"name": "bull", "index": 2}`. Based on the most recent bar. |
| `next_state_probabilities` | object | `{"bear": 0.1, "sideways": 0.3, "bull": 0.6}`. From transition matrix row for the current regime. |
| `signal` | float | `P(bull) - P(bear)`, range [-1, 1]. Positive = bullish signal. |
| `transition_matrix` | 3×3 array | Row-normalised transition matrix. Row `i` = probabilities of transitioning *from* state `i`. |
| `persistence_diagonal` | object | `{"bear": 0.85, ...}`. Diagonal of transition matrix — higher = stickier regime. |
| `stationary_distribution` | object | `{"bear": 0.25, ...}`. Long-run fraction of time spent in each regime. |
| `walk_forward` | object | `{"sharpe": 1.2, "max_drawdown": -0.15, "n_trades": 45}`. NaN values become `null` in JSON. |
| `hmm` | object | `{"available": true/false, ...}`. HMM results if available. |
| `hmm_test_extras` | object | `{"n_states": 3, "method": "threshold", "data_points": 2528, "regime_counts": {...}}`. Skill-specific metadata. |
| `forecast` | object | 1, 5, and 20-step-ahead regime probability forecasts via matrix exponentiation. |
| `framework` | str | `"hmm_test v0.1.0"`. |
| `disclaimer` | str | Standard disclaimer. |

### hmm_test_extras structure

```json
{
  "n_states": 3,
  "method": "threshold",
  "data_points": 2528,
  "regime_counts": {
    "bear": 500,
    "sideways": 1000,
    "bull": 1028
  }
}
```

## Processing Pipeline

1. **Load data** — CSV via auto-detection of date/close columns, or yfinance ticker download.
2. **Compute returns** — `pct_change().dropna()`.
3. **Classify regimes** — Rolling sum of returns over `--window` bars vs `--threshold`.
4. **Build transition matrix** — Counts → row-normalised 3×3 probability matrix.
5. **Compute statistics** — Stationary distribution, persistence diagonal, directional signal.
6. **Walk-forward backtest** — No-lookahead: at each bar `t`, uses only data `[0:t)` for regime classification, then trades at `t` using the signal. Equity curve built from daily P&L.
7. **HMM analysis** (optional) — Fits a GaussianHMM with `n_states=3`, re-orders states by mean return (lowest→bear, highest→bull), returns labeled regimes and transition matrix.
8. **Forecast** — Matrix exponentiation to project regime probabilities 1, 5, and 20 steps ahead.

## Modes

### Threshold (default, always runs)

Fast, deterministic. Good for quick regime checks. Uses rolling return sums. No model fitting required.

### HMM (optional, runs when `--hmm` or default)

Slower but richer. Fits a Gaussian Hidden Markov Model with 3 states. States are labeled post-hoc by ascending mean return expectation. **Caveat**: labels may swap on re-fit — agents should not rely on state index stability across runs.

If HMM fails (e.g. hmmlearn not installed, insufficient data, convergence failure), the output includes `{"available": false, "reason": "..."}` and the threshold-based results are still valid.

## Walk-Forward Backtest Detail

- **No lookahead bias**: At time `t`, only price data up to `t-1` is used for regime classification.
- **Signal trading**: Position = `clip(signal, -1, 1)` applied at bar `t`. No transaction costs modeled.
- **Equity curve**: Cumulative product of `1 + position * return`.
- **Min train**: First `min_train` bars are excluded from trading (insufficient history).
- **Sharpe ratio**: Annualised using `sqrt(252)` daily factor. Returns `NaN` (→ `null` in JSON) if insufficient data.
- **Drawdown**: Max drawdown from peak equity. Negative value (e.g. `-0.15` = 15% drawdown).

## Signal Interpretation

```
signal = P(next_regime = Bull) - P(next_regime = Bear)
```

- **+1.0**: Strong bullish expectation → go long.
- **0.0**: Neutral → stay flat.
- **-1.0**: Strong bearish expectation → go short.
- Intermediate values: fractional positions (e.g. `0.5` = half long).

## Composition Patterns

### Consuming the skill from another agent

```python
import subprocess, json

result = subprocess.run(
    ["python", "scripts/cli.py", "--csv", "data.csv", "--json"],
    capture_output=True, text=True
)

if result.returncode == 0:
    data = json.loads(result.stdout)
    regime = data["current_regime"]["name"]
    signal = data["signal"]
    sharpe = data["walk_forward"]["sharpe"]
    # Use regime + signal for allocation decision
else:
    error = json.loads(result.stdout)["error"]
    # Handle failure
```

### Ticker scanning

Loop over tickers and collect signals:

```bash
for ticker in SPY QQQ IWM DIA; do
  python scripts/cli.py --ticker $ticker --json --no-hmm
done
```

### Gate a strategy with regime

Use `signal > 0` as a long-only filter, or `abs(signal) > 0.2` as a minimum conviction threshold.

## Gotchas

1. **HMM label instability**: Labels "bear"/"bull" are inferred from ascending state means and can swap between re-fits. Threshold mode labels are stable.
2. **Sharpe for intraday data**: The default `sqrt(252)` assumes daily bars. For intraday data, the Sharpe may be inflated. Use `--min-train` appropriately.
3. **Insufficient data**: If `len(prices) < min_train + 1`, walk_forward returns `null` Sharpe and 0 trades.
4. **CSV format**: Auto-detects date and close columns. If detection fails, specify columns explicitly or reformat the CSV.
5. **yfinance dependency**: Optional. Install with `pip install yfinance` or `uv sync --extra yfinance`.
6. **signal = 0**: Can happen when bull and bear probabilities are equal. Common in sideways markets.
7. **Threshold sensitivity**: Small `--threshold` values produce frequent regime switches. Large values make the regime "sticky".

## Reference Index

For deeper dives on specific topics, see the reference files:

| Topic | File |
|-------|------|
| HMM theory, EM algorithm, Viterbi, state ordering | [`references/hmm_theory.md`](references/hmm_theory.md) |
| Feature engineering (log-returns, ATR, ROC, volatility) | [`references/feature_engineering.md`](references/feature_engineering.md) |
| Walk-forward methodology, drawdown, position sizing | [`references/backtesting_detail.md`](references/backtesting_detail.md) |
| All parameters, tuning, recommended values by asset class | [`references/configuration.md`](references/configuration.md) |
| Common errors and fixes | [`references/troubleshooting.md`](references/troubleshooting.md) |

## Dependencies

Core: `numpy`, `pandas`, `scikit-learn`, `scipy`, `hmmlearn`
Optional: `yfinance` (for `--ticker`)

All core dependencies are listed in `pyproject.toml` and managed with `uv`.
