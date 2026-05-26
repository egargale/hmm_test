---
name: hmm-regime-detection
description: >
  Detect the current market regime (Bull/Bear/Sideways) for any asset using
  threshold-based classification or Hidden Markov Models. Three independent
  engines: threshold (fast, close-only), messina (HMM + 19 Messina features),
  hmm (HMM + ~50 generic features). Each engine produces a self-contained
  output with bias-free walk-forward backtest and trade-level analytics.
  Use when the user wants regime detection, Markov transition analysis,
  walk-forward backtesting, or regime-based risk gating — on a ticker
  (via yfinance) or on CSV price data.
---

# HMM Regime Detection Skill

Detect whether an asset is in a **Bull** (uptrend), **Bear** (downtrend), or **Sideways** (range-bound) regime. Three independent engines are available, each producing a complete, self-contained analysis with bias-free walk-forward backtest and trade-level analytics.

## Quick Invocation

```bash
SKILL_DIR="$SKILL_DIR"

# ── NORMAL RUNS ──────────────────────────────────────────────────────────
"$SKILL_DIR/run.sh" --ticker KO --json
"$SKILL_DIR/run.sh" --csv data.csv --json
"$SKILL_DIR/run.sh" --csv data.csv --json --engine messina
```

> The `run.sh` wrapper is self-bootstrapping — it creates a venv and installs
> dependencies on first run. No manual setup required.

## Engines

Three independent, self-contained engines. Pick one per invocation via `--engine`:

| Engine | `--engine` | Features | Model | Data required |
|--------|-----------|----------|-------|---------------|
| **Threshold** | `threshold` (default) | 1 (returns) | Rolling return vs. threshold | Close prices |
| **Messina** | `messina` | 19 (Wilder's) | GaussianHMM on expanding window | OHLCV |
| **HMM** | `hmm` | ~50 (SMA-based) | GaussianHMM on expanding window | OHLCV |

The Messina and HMM engines require OHLCV data and automatically receive it when using `--ticker`. For CSV mode, the CSV must contain open/high/low/close/volume columns.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv` | path | — | Path to CSV file with price data. |
| `--ticker` | str | — | yfinance ticker (e.g. `ES=F`, `SPY`, `BTC-USD`). |
| `--json` | flag | off | Output JSON to stdout. On error: `{"error": "..."}` + exit 1. |
| `--engine` | str | `threshold` | One of `threshold`, `messina`, `hmm`. |
| `--window` | int | 20 | Rolling window for regime classification. |
| `--threshold` | float | 0.05 | Return threshold for bull/bear classification. |
| `--min-train` | int | 252 | Minimum bars before walk-forward trading starts. |
| `--n-states` | int | 3 | Number of HMM states (ignored by threshold engine). |

### Removed flags (v0.2.0)

`--hmm`/`--no-hmm` and `--messina` have been replaced by `--engine`. Old invocations will error.

## State Interpretation

| Regime | Index | Position |
|--------|-------|----------|
| **Bear** | 0 | Short (−1) |
| **Sideways** | 1 | Flat (0) |
| **Bull** | 2 | Long (+1) |

The threshold engine uses rolling return to classify: above `+threshold` = Bull, below `-threshold` = Bear, otherwise Sideways. HMM engines sort states by ascending mean return to determine the mapping.

## JSON Output Contract

```json
{
  "source": "ES=F",
  "engine": "threshold",
  "dates": {"start": "2016-01-04", "end": "2026-05-21"},
  "current_regime": {"name": "bull", "index": 2},
  "signal": 0.62,
  "next_state_probabilities": {"bear": 0.08, "sideways": 0.30, "bull": 0.62},
  "transition_matrix": [[0.88, 0.07, 0.05], [0.12, 0.70, 0.18], [0.04, 0.19, 0.77]],
  "stationary_distribution": {"bear": 0.22, "sideways": 0.35, "bull": 0.43},
  "persistence_diagonal": {"bear": 0.88, "sideways": 0.70, "bull": 0.77},
  "regime_counts": {"bear": 480, "sideways": 760, "bull": 960},
  "walk_forward": {
    "sharpe": 0.51,
    "max_drawdown": -0.15,
    "n_trades": 42,
    "win_rate": 0.57,
    "profit_factor": 1.80,
    "total_return": 0.23
  },
  "forecast": {
    "1_step":  {"bear": 0.08, "sideways": 0.30, "bull": 0.62},
    "5_step":  {"bear": 0.12, "sideways": 0.33, "bull": 0.55},
    "20_step": {"bear": 0.18, "sideways": 0.35, "bull": 0.47}
  },
  "engine_info": {
    "method": "threshold",
    "features": "returns",
    "n_states": 3
  },
  "framework": "hmm_test v0.2.0",
  "disclaimer": "Regime detection is probabilistic. Past transitions do not guarantee future regimes. Not financial advice."
}
```

### Walk-forward fields

All values are floats (may be `null` for insufficient data):

| Field | Description |
|-------|-------------|
| `sharpe` | Annualised Sharpe ratio (daily factor √252) |
| `max_drawdown` | Maximum drawdown from peak (negative, e.g. −0.15 = 15%) |
| `n_trades` | Number of completed trades (integer) |
| `win_rate` | Fraction of trades with positive P&L |
| `profit_factor` | Gross profit / gross loss |
| `total_return` | Total return over the trading period |

### Engine info fields

| Field | Description |
|-------|-------------|
| `method` | Engine name: `threshold`, `messina`, or `hmm` |
| `features` | Feature mode: `returns`, `messina`, or `generic` |
| `n_states` | Number of HMM states (always 3, included for reference) |
| `caveat` | Present on HMM engines: warns about label instability on re-fit |

## Processing Pipeline

1. **Load data** — CSV via auto-detection of date/close columns, or yfinance ticker download. For `--engine messina`/`hmm`, full OHLCV is loaded.
2. **Compute returns** — `pct_change().dropna()`.
3. **Classify regimes** — Rolling sum of returns over `--window` bars vs `--threshold`.
4. **Build transition matrix** — Counts → row-normalised 3×3 probability matrix.
5. **Compute statistics** — Stationary distribution, persistence diagonal, directional signal.
6. **Walk-forward backtest** — No-lookahead: at each bar `t`, regime classification uses only data `[0:t)`. Discrete positions (−1, 0, +1) via `{bear: short, sideways: flat, bull: long}`. Trades fire on regime changes. Daily P&L from lagged positions × returns.
7. **Forecast** — Matrix exponentiation to project regime probabilities 1, 5, and 20 steps ahead.

## Signal Interpretation

```
signal = P(next_regime = Bull) - P(next_regime = Bear)
```

- **+1.0**: Strong bullish expectation → long conviction
- **0.0**: Neutral → flat
- **-1.0**: Strong bearish expectation → short conviction
- Intermediate values: fractional between 0.0 and extremes

## Composition Patterns

### Consuming the skill from another agent

```python
import subprocess, json

result = subprocess.run(
    ["./run.sh", "--csv", "data.csv", "--json", "--engine", "threshold"],
    capture_output=True, text=True
)

if result.returncode == 0:
    data = json.loads(result.stdout)
    regime = data["current_regime"]["name"]
    signal = data["signal"]
    sharpe = data["walk_forward"]["sharpe"]
    win_rate = data["walk_forward"]["win_rate"]
    # Use regime + signal for allocation decision
else:
    error = json.loads(result.stdout)["error"]
```

### Ticker scanning

```bash
for ticker in SPY QQQ IWM DIA; do
  ./run.sh --ticker $ticker --json --engine threshold
done
```

## Gotchas

1. **HMM label instability**: Messina/hmm engines sort states by mean return. Labels may swap between re-fits on different data. Threshold engine labels are stable.
2. **HMM engines need OHLCV**: `--csv` with `--engine messina` or `--engine hmm` requires open/high/low/close/volume columns. For close-only CSVs, use `--engine threshold`.
3. **Sharpe for intraday data**: The default `sqrt(252)` assumes daily bars. For intraday data, the Sharpe may be inflated.
4. **Insufficient data**: If `len(prices) < min_train + 1`, walk_forward returns `null` for all fields except `n_trades` (0).
5. **CSV format**: Auto-detects date and close columns. If detection fails, specify columns explicitly or reformat the CSV.
6. **yfinance dependency**: Optional. `run.sh` installs it automatically. For manual install: `uv sync --extra yfinance`.
7. **signal = 0**: Can happen when bull and bear probabilities are equal. Common in sideways markets.
8. **Threshold sensitivity**: Small `--threshold` values produce frequent regime switches. Large values make the regime "sticky".

## Reference Index

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
