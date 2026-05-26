# hmm-regime-detection

Detect market regimes (Bull/Bear/Sideways) for any asset using Hidden Markov Models and Markov chain analysis.

## Installation

This is an **Agent Skill** for LLM coding agents (Claude, GPT, etc.). Copy or symlink it into your agent's `skills/` folder:

```bash
# Claude Code
cp -r /path/to/hmm_test ~/.claude/skills/hmm-regime-detection

# Or symlink for live updates
ln -s /path/to/hmm_test ~/.claude/skills/hmm-regime-detection
```

Install Python dependencies:

```bash
cd hmm_test
uv sync
```

## What It Does

Given a ticker or CSV with price data, it outputs a structured JSON with:

- **Current regime** — Bull, Bear, or Sideways
- **Transition matrix** — probabilities of switching between regimes
- **Signal** — `P(bull) - P(bear)` in `[-1, 1]`
- **Walk-forward backtest** — Sharpe, max drawdown, trade count
- **Forecasts** — 1-step, 5-step, 20-step regime probabilities
- **Three independent engines** — threshold (close-only), messina (HMM + 19 features), hmm (HMM + ~50 features)

## Usage

```bash
# From CSV (threshold engine, default)
./run.sh --csv data.csv --json

# From ticker (needs yfinance, installed automatically by run.sh)
./run.sh --ticker ES=F --json

# Messina engine (HMM with 19 Wilder's-smoothed features, requires OHLCV)
./run.sh --csv data.csv --json --engine messina

# Generic HMM engine (~50 SMA-based features, requires OHLCV)
./run.sh --ticker SPY --json --engine hmm

# Custom parameters
./run.sh --csv data.csv --json --window 10 --threshold 0.03
```

## Output Contract

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
    "sharpe": 0.51, "max_drawdown": -0.15, "n_trades": 42,
    "win_rate": 0.57, "profit_factor": 1.80, "total_return": 0.23
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

## Full Documentation

See [SKILL.md](SKILL.md) for the complete agent-facing skill definition, arguments, contract schema, composition patterns, and gotchas. See [references/](references/) for deep dives on HMM theory, feature engineering, backtesting, configuration, and troubleshooting.

## License

MIT
