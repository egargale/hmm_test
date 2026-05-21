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
- **Optional HMM** — Gaussian/GMM hidden Markov model for richer state detection

## Usage

```bash
# From CSV
PYTHONPATH=scripts python scripts/cli.py --csv data.csv --json --no-hmm

# From ticker (needs yfinance)
PYTHONPATH=scripts python scripts/cli.py --ticker ES=F --json

# Custom parameters
PYTHONPATH=scripts python scripts/cli.py --csv data.csv --json --window 10 --threshold 0.03
```

## Output Contract

```json
{
  "source": "ES=F",
  "current_regime": {"name": "bull", "index": 2},
  "signal": 0.45,
  "next_state_probabilities": {"bear": 0.10, "sideways": 0.30, "bull": 0.60},
  "transition_matrix": [[0.85, 0.10, 0.05], [0.08, 0.72, 0.20], [0.05, 0.15, 0.80]],
  "stationary_distribution": {"bear": 0.25, "sideways": 0.40, "bull": 0.35},
  "walk_forward": {"sharpe": 1.2, "max_drawdown": -0.15, "n_trades": 45},
  "hmm": {"available": false}
}
```

## Full Documentation

See [SKILL.md](SKILL.md) for the complete agent-facing skill definition, arguments, contract schema, composition patterns, and gotchas. See [references/](references/) for deep dives on HMM theory, feature engineering, backtesting, configuration, and troubleshooting.

## License

MIT
