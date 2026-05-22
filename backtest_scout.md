# Backtesting Architecture Scout Report

## Files Examined

| File | Role |
|------|------|
| `scripts/regime/walk_forward.py` | Walk-forward backtest (threshold-based only) |
| `scripts/regime/pipeline.py` | Orchestration pipeline that calls `walk_forward_backtest()` |
| `scripts/backtesting/__init__.py` | Re-exports 7 functions from the module |
| `scripts/backtesting/strategy_engine.py` | HMM-aware backtest engine (unused) |
| `scripts/backtesting/performance_metrics.py` | Metric calculators (Sharpe, drawdown, etc.) |
| `scripts/backtesting/performance_analyzer.py` | Wraps metrics + trade analysis |
| `scripts/backtesting/bias_prevention.py` | Lookahead bias detection utilities |
| `scripts/backtesting/utils.py` | Validation, alignment, regime analysis helpers |
| `scripts/regime/hmm_adapter.py` | HMM state sequence generator (no backtest integration) |
| `scripts/regime/markov_chain.py` | Threshold-based classification + signal functions |
| `scripts/utils/data_types.py` | Core dataclasses: `Trade`, `BacktestResult`, `PerformanceMetrics`, `BacktestConfig` |
| `scripts/utils/config.py` | Pydantic `BacktestConfig` (duplicate of dataclass) |
| `references/backtesting_detail.md` | Docs describing the walk-forward approach |

---

## 1. Is `walk_forward_backtest()` hardcoded to threshold-based signals?

**Yes, completely hardcoded.** Lines 35-46 of `walk_forward.py`:

```python
hist_returns = returns.iloc[:t]
regimes = classify_regimes(hist_returns, window=window, threshold=threshold)
transmat = build_transition_matrix(regimes)
current_regime = int(regimes[-1])
next_state_probs = transmat[current_regime]
signal = compute_signal(next_state_probs)
positions[t] = np.clip(signal, -1.0, 1.0)
```

There is **no injection point** for external regime sequences or signals. The function:
- Only accepts `prices`, `window`, `threshold`, `min_train` — no `states` or `signals` parameter.
- Calls `classify_regimes()` from `markov_chain.py` which is a **threshold-based** rolling-sum classifier (not HMM).
- Computes a continuous `[-1, 1]` signal, not discrete state-to-position mapping.
- Uses a simplified P&L model: `position * return[t]` with no transaction costs.

**Bottom line:** It cannot accept HMM states or external signals without modification.

---

## 2. What does `scripts/backtesting/strategy_engine.py` do? Is it used anywhere?

**What it does:**
- `backtest_strategy(states, prices, config)` — the only function in the codebase designed to accept HMM state sequences (`np.ndarray`) and map them to positions via `config.state_map` (a `Dict[int, int]` like `{0: 1, 1: 0, 2: -1}`).
- `backtest_with_analysis(states, prices, config)` — wraps the above, creates equity curve, analyzes positions, returns `BacktestResult`.
- `backtest_strategy()` includes:
  - **Lookahead bias prevention** via `np.roll(states, 1)` (lines 172-178)
  - **Transaction costs**: commission (fixed per trade) and slippage (bps)
  - **Trade-level logging** using the `Trade` dataclass (entry/exit, P&L, costs)
  - **Validates** input with `validate_backtest_inputs()` (length matching, state range)

**Is it used anywhere?** **No.** The only call site is `backtest_with_analysis()` calling `backtest_strategy()` internally (line 460). No test, pipeline, CLI, or example file calls either function. The `__init__.py` re-exports them, but nothing imports them from there.

**Why it's strategically important:** This is the **ready-made bridge** for HMM-driven backtesting. It requires zero new code to accept HMM states.

---

## 3. What does `scripts/backtesting/performance_metrics.py` expose?

| Function | Signature | Purpose |
|----------|-----------|---------|
| `calculate_sharpe_ratio()` | `(equity_curve, risk_free_rate=0.02, frequency=None)` | Sharpe with auto frequency inference, `sqrt(annualization)` scaling |
| `calculate_drawdown_metrics()` | `(equity_curve)` | Returns dict: `max_drawdown`, `max_drawdown_duration`, `avg_drawdown`, `avg_recovery_time` |
| `calculate_performance()` | `(equity_curve, risk_free_rate=0.02, benchmark_curve=None, frequency=None)` | Returns `PerformanceMetrics` dataclass with all core metrics |
| `calculate_annualized_return()` | `(equity_curve, frequency=None)` | CAGR from first/last equity values |
| `calculate_annualized_volatility()` | `(returns, frequency=None)` | `std(returns) * sqrt(annualization_factor)` |
| `calculate_calmar_ratio()` | `(equity_curve, risk_free_rate=0.02)` | CAGR / |max drawdown| |
| `infer_trading_frequency()` | `(equity_curve)` | Auto-detects hourly/daily/weekly/monthly from index diffs |
| `get_annualization_factor()` | `(frequency)` | Maps frequency string to float (252 for daily, etc.) |
| `calculate_returns()` | `(equity_curve, frequency=None)` | `pct_change()` wrapper |
| `validate_performance_metrics()` | `(metrics)` | Sanity checks on metric values |
| `create_performance_summary()` | `(metrics)` | Human-readable formatted string |

**Key difference from `walk_forward.py`:** The walk-forward version computes Sharpe using `252*78` multiplier on raw returns (intraday assumption). The `performance_metrics.py` version uses the standard `sqrt(annualization)` approach with auto frequency detection.

---

## 4. Any sign of an HMM-driven backtest anywhere?

**None.** The search covered:
- All imports in `pipeline.py`, `cli.py`, and tests
- All `from backtesting import ...` statements
- All calls to `backtest_strategy()`, `backtest_with_analysis()`
- All references to `state_map` or `BacktestConfig`
- The full `scripts/regime/` and `scripts/backtesting/` directories

The HMM adapter (`hmm_adapter.py`) produces regime labels via `run_hmm_regime()`, but **the returned dict with `"regimes"` is never fed into any backtesting function**. The pipeline runs both threshold backtest and HMM analysis, but they're entirely independent paths:

```
pipeline.run()
├── classify_regimes() → walk_forward_backtest()  ← threshold only
└── run_hmm_regime()   → returns dict with regimes  ← never backtested
```

---

## 5. Useful abstractions in `bias_prevention.py` and `utils.py`

### `bias_prevention.py`

| Component | What it does |
|-----------|-------------|
| `detect_lookahead_bias()` | Comprehensive check: timing consistency + feature availability + position shifting |
| `validate_timing_consistency()` | Checks decisions use `lag_periods` old states |
| `validate_position_shifting()` | Verifies positions match expected positions from `state_map` + 1-bar shift |
| `apply_bias_prevention()` | Lags states and positions by N periods |
| `BiasDetectionResult` dataclass | Structured output: violations, risk score, recommendations |

### `utils.py`

| Function | What it does |
|----------|-------------|
| `validate_backtest_inputs()` | Ensures states/prices lengths match, config is valid |
| `calculate_transaction_costs()` | Commission + slippage calculation |
| `calculate_position_returns()` | `lagged_positions * price_returns * position_size` |
| `analyze_regime_performance()` | Per-state return/volatility/win-rate breakdown |
| `calculate_rolling_regime_metrics()` | Rolling window analysis per state |
| `create_trade_log_dataframe()` | Converts `List[Trade]` to `pd.DataFrame` |
| `create_sample_price_data()` | Synthetic price generation for testing |
| `create_sample_state_sequence()` | Synthetic state sequence generation |
| `align_state_and_price_data()` | Length alignment strategies |

### `performance_analyzer.py`

| Function | What it does |
|----------|-------------|
| `analyze_performance()` | Wraps `calculate_performance()` + trade-based metrics (win rate, profit factor, Sortino) |
| `analyze_trade_distribution()` | P&L distribution, duration analysis |
| `benchmark_comparison()` | Correlation, beta, alpha, tracking error vs benchmark |

---

## Key Architectural Insight: Two Disconnected Backtest Worlds

```
┌────────────────────────────────────────────────────┐
│             walk_forward_backtest()                │
│  (scripts/regime/walk_forward.py)                  │
│                                                    │
│  Input: prices, window, threshold, min_train       │
│  Signal: threshold-based rolling-sum classifier    │
│  Positions: continuous [-1, 1] from P(bull)-P(bear)│
│  Costs: none                                       │
│  Lookahead: no shift (uses hist-only returns)      │
│  Output: {sharpe, max_drawdown, n_trades} dict     │
│                                                    │
│  ● USED by pipeline.py and cli.py                  │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│      backtest_strategy(states, prices, config)     │
│  (scripts/backtesting/strategy_engine.py)          │
│                                                    │
│  Input: states (np.ndarray), prices, config        │
│  Signal: state_map lookup {state: position}        │
│  Positions: discrete {-1, 0, 1}                    │
│  Costs: commission + slippage                      │
│  Lookahead: np.roll(states, 1)                     │
│  Output: (positions_series, trades_list) tuple     │
│                                                    │
│  ● NEVER CALLED anywhere                           │
└────────────────────────────────────────────────────┘
```

---

## Minimal Change to Enable HMM-Driven Backtest Comparison

**Estimated effort: ~10-15 lines of new code, no changes to existing files.**

Write a new thin script (e.g., `scripts/backtesting/hmm_backtest.py`) that does:

```python
# 1. Run HMM to get states
from regime.hmm_adapter import run_hmm_regime
hmm_result = run_hmm_regime(prices, n_states=3)
if not hmm_result["available"]:
    raise RuntimeError(hmm_result["reason"])

# 2. Convert regime labels to integer states
state_to_int = {"bear": 0, "sideways": 1, "bull": 2}
states = np.array([state_to_int[r] for r in hmm_result["regimes"]], dtype=int)

# 3. Truncate to price length (HMM may drop NaN rows)
states, prices_aligned = align_state_and_price_data(states, prices)

# 4. Run backtest
from backtesting.strategy_engine import backtest_strategy
from utils.data_types import BacktestConfig
config = BacktestConfig(
    state_map={0: -1, 1: 0, 2: 1},   # bear → short, sideways → flat, bull → long
    commission_per_trade=0.0,
    slippage_bps=0.0,
)
positions, trades = backtest_strategy(states, prices_aligned, config)

# 5. Measure performance on the same metric contract
from backtesting.performance_metrics import calculate_performance
equity = (1 + positions.shift(1) * prices_aligned.pct_change()).cumprod()
metrics = calculate_performance(equity)
```

### Why it's this simple

1. **`backtest_strategy()` already accepts HMM states** as `np.ndarray` — its signature and docstring say "HMM state sequence".
2. **`BacktestConfig.state_map`** is the only bridge needed to map HMM state IDs (0,1,2) to positions (-1,0,1).
3. **Lookahead bias prevention** is built-in via `np.roll(states, 1)`.
4. **Performance metrics** (`calculate_performance`, `calculate_sharpe_ratio`, `calculate_drawdown_metrics`) are shared between both paths.
5. Unlike `walk_forward_backtest()`, this is a **simple vectorized backtest** (single-pass over states), not a walk-forward recomputation — appropriate because HMM states are already fit on all available data.

### What the adapter must handle

| Challenge | Mitigation |
|-----------|-----------|
| HMM may produce fewer states than price rows (NaN drop) | `align_state_and_price_data()` in `utils.py` |
| HMM label order may swap on re-fit | `run_hmm_regime()` already sorts states by ascending mean return |
| Lookahead bias from full-sample HMM fit | Must acknowledge this in documentation; use walk-forward HMM refitting for purist comparison |

### Recommended file layout

```
scripts/backtesting/
├── hmm_backtest.py          ← NEW: adapter, imports strategy_engine
```

Or simpler: one block in `scripts/regime/pipeline.py` that, when `use_hmm=True`, computes HMM backtest metrics alongside threshold metrics and includes both in the return dict.
