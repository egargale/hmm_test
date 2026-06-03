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
- **Walk-forward backtest** — Sharpe, max drawdown, trade count, win rate, profit factor
- **Forecasts** — 1-step, 5-step, 20-step regime probabilities
- **Duration forecast** — expected remaining bars of current regime via Weibull or Cox survival analysis
- **Five independent engines** — threshold (close-only), messina (HMM + 19 features), hmm (HMM + ~50 features), robust_hmm (outlier-resistant), fshmm (feature saliency HMM)

## Usage

```bash
# From CSV (threshold engine, default)
./scripts/run.sh --csv data.csv --json

# From ticker (needs yfinance, installed automatically by run.sh in scripts/)
./scripts/run.sh --ticker ES=F --json

# Messina engine (HMM with 19 Wilder's-smoothed features, requires OHLCV)
./scripts/run.sh --csv data.csv --json --engine messina

# Generic HMM engine (~50 SMA-based features, requires OHLCV)
./scripts/run.sh --ticker SPY --json --engine hmm

# Robust HMM engine (outlier-resistant with Huber IRLS or MCD)
./scripts/run.sh --csv data.csv --json --engine robust_hmm --robust-method mcd

# Feature Saliency HMM engine (learns per-feature relevance weights)
./scripts/run.sh --ticker SPY --json --engine fshmm --saliency-threshold 0.5

# Duration forecasting (survival analysis on regime spell lengths)
./scripts/run.sh --ticker SPY --json --engine messina --duration-forecast

# Custom parameters
./scripts/run.sh --csv data.csv --json --window 10 --threshold 0.03
```

## Output Contract

```json
{
  "source": "ES=F",
  "engine": "fshmm",
  "dates": {"start": "2016-05-31", "end": "2026-05-29"},
  "current_regime": {"name": "bull", "index": 2},
  "signal": 0.82,
  "next_state_probabilities": {"bear": 0.02, "sideways": 0.16, "bull": 0.82},
  "transition_matrix": [[0.85, 0.10, 0.05], [0.03, 0.92, 0.05], [0.01, 0.07, 0.92]],
  "stationary_distribution": {"bear": 0.10, "sideways": 0.45, "bull": 0.45},
  "persistence_diagonal": {"bear": 0.85, "sideways": 0.92, "bull": 0.92},
  "regime_counts": {"bear": 200, "sideways": 1200, "bull": 1100},
  "walk_forward": {
    "sharpe": 0.65,
    "max_drawdown": -0.24,
    "n_trades": 20,
    "win_rate": 0.55,
    "profit_factor": 4.88,
    "total_return": 2.07
  },
  "forecast": {
    "1_step":  {"bear": 0.02, "sideways": 0.16, "bull": 0.82},
    "5_step":  {"bear": 0.07, "sideways": 0.28, "bull": 0.65},
    "20_step": {"bear": 0.15, "sideways": 0.42, "bull": 0.43}
  },
  "engine_info": {
    "method": "fshmm",
    "features": "generic",
    "n_states": 3,
    "warmup_bars": 252,
    "caveat": "HMM states sorted by mean return; labels may swap on re-fit",
    "feature_saliency": [0.85, 0.92, 0.32, 0.11, 0.88],
    "selected_features": ["log_ret", "rsi_14", "sma_50"]
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
  "regime_transitions": [
    {"date": "2026-03-15", "from_regime": "sideways", "to_regime": "bull", "bar_index": 2450},
    {"date": "2026-01-10", "from_regime": "bear", "to_regime": "sideways", "bar_index": 2398}
  ],
  "framework": "hmm_regime_detection v0.5.0",
  "disclaimer": "Regime detection is probabilistic. Past transitions do not guarantee future regimes. Not financial advice."
}
```

## Full Documentation

See [SKILL.md](SKILL.md) for the complete agent-facing skill definition, arguments, contract schema, composition patterns, and gotchas. See [references/](references/) for deep dives on HMM theory, feature engineering, backtesting, configuration, and troubleshooting.

## License

MIT
