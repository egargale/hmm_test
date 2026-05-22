# Three independent regime-detection engines

The HMM analysis was misused — a full-dataset GaussianHMM fit with lookahead
bias, never connected to the walk-forward backtest, producing an `hmm.regimes[]`
array that nothing downstream consumed. We decided to restructure into three
independent, self-contained engines rather than trying to bolt HMM signals
into the existing threshold pipeline.

## Considered options

### A) HMM as threshold replacement (original handoff plan)

Make `--hmm` swap the walk-forward signal source from threshold to HMM.
Single output block, single backtest, but the signal origin changes.

**Rejected** because: (1) the two signal models compute fundamentally
different things — threshold is `rolling_sum(returns) > limit`, HMM is
a full latent-state model — so the walk_forward block would produce
incomparable results depending on a flag marketed as "optional enrichment";
(2) the HMM `regimes[]` array remained unused for anything besides the
`hmm` JSON block; (3) the old architecture produced two different
transition matrices in one output, confusing consumers.

### B) HMM as separate supplementary analysis

Keep the threshold track as the single source of truth, and produce an
HMM block alongside for informational purposes only.

**Rejected** because: the HMM block was never actionable — `regimes[]`
with full-sample lookahead bias can't be traded, and the block had no
connection to any P&L model.

### C) Three independent engines (chosen)

Each engine is a complete, self-contained analysis: regime classification,
transition matrix, stationarity, forecasts, and a bias-free walk-forward
backtest with discrete trade-level analytics. The user selects one per
invocation via `--engine threshold|messina|hmm`. Engines are never
compared within a single run — no cross-engine Sharpe comparison,
no combined output.

**Chosen because**: (1) eliminates lookahead bias by design — each
engine owns its full pipeline and must be bias-free; (2) all outputs
are actionable — the walk_forward block always comes from the same
engine that produced the regime labels; (3) same output schema across
engines means consumers read `result["walk_forward"]["sharpe"]`
regardless of engine; (4) the three engines genuinely differ —
threshold is fast and works on close-only data, messina uses 12
Wilder's-smoothed features, HMM uses ~44 SMA-based features —
and each produces a realizable trading strategy.

## Consequences

- **Harder to compare engines.** A consumer who wants threshold vs. HMM
  backtest numbers must run the tool twice and compare manually. This is
  intentional — a single run should produce a coherent strategy, not a
  comparison matrix.
- **Backward-incompatible.** Old `--hmm`/`--no-hmm`/`--messina` flags
  removed. Old `hmm` and `hmm_test_extras` JSON keys removed. SKILL.md
  bumped to v0.2.0.
- **HMM engines require OHLCV.** `--csv` with `--engine hmm` or
  `--engine messina` demands open/high/low/close/volume columns.
  Close-only CSVs work only with `--engine threshold`.
- **Walk-forward HMM performance.** Fitting a GaussianHMM on expanding
  windows with ~44 features is O(n × m). Skip-N refitting (~100 fits
  for ~2000 bars) balances accuracy with ~60s wall time. Fully
  per-bar refitting would be ~10× slower.
- **Orphaned `backtesting/strategy_engine.py`.** Deleted. The discrete
  trade-booking logic was absorbed into `regime/walk_forward.py`.
