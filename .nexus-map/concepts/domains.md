> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: Derived from CONTEXT.md and AST analysis

# Core Domain Concepts

## Engine
A self-contained regime classifier implementing the `RegimeEngine` Protocol (`precompute` → `classify` → `run_classify`). Five exist: threshold (close-only, vectorized), hmm (HMM + ~50 generic features), messina (HMM + 19 Wilder's features), robust_hmm (HMM + Huber/MCD outlier resistance), fshmm (HMM + feature saliency learning). Each engine produces `ClassifyOutput` consumed by the pipeline.

## Regime (Market Regime)
The market condition at a point in time: **Bear (0)** = declining, **Sideways (1)** = range-bound, **Bull (2)** = rising. States map to trading positions: 0 → Short (-1), 1 → Flat (0), 2 → Long (+1). The threshold engine is the only one working with close-only data; all HMM-family engines require OHLCV.

## HMM Latent State
Raw state index (0, 1, ..., n) output by `GaussianHMM.predict()`. These indices are arbitrary and must be mapped to regimes by sorting ascending mean return: lowest → Bear, middle → Sideways, highest → Bull. BIC selection (`--n-states auto`) can find the optimal count.

## Walk-forward Backtest
A bias-free backtest where at each bar t, only data up to t-1 is used for classification. Position at bar t is determined by the regime at t-1. Produces discrete trades with Sharpe, max drawdown, win rate, profit factor. Optional filters: dwell-time (N consecutive bars) and hysteresis (posterior margin).

## Signal
`P(next_regime = Bull) - P(next_regime = Bear)` in range [-1, 1]. Derived from the transition matrix row for the current regime. Positive = bullish outlook, negative = bearish.

## Transition Matrix
A 3×3 row-normalized matrix where T[i][j] = probability of i→j transition. Row sums to 1.0. Diagonal entries measure regime stickiness/persistence.

## Degenerate Auto-recovery
When an HMM engine's pre-check detects a 3-state fit collapsing (one state < 5% of bars), the engine auto-downgrades to n_states=2. Mid-stream degeneration (detected during walk-forward classify) also triggers downgrade. The original requested n_states and recovery event are recorded in engine_info.

## PCA Whitening
Optional dimensionality reduction applied after z-score normalization and before `model.fit()`. Configured per-engine via `pca_variance` parameter. Component count is determined on first refit and reused thereafter.

## Duration Forecasting
Survival analysis on regime spell lengths using Weibull or Cox models. Outputs expected remaining bars, hazard rate, median survival time, and distribution shape/scale parameters.

## BIC State Selection
Automatic selection of HMM state count via Bayesian Information Criterion. Fits GaussianHMM for k in [2, max_states] with multiple restarts, returns count with lowest BIC. Default remains `n_states=3`.
