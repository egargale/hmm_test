# HMM Regime Detection

Detect market regimes (Bull/Bear/Sideways) using threshold-based classification or Hidden Markov Models, with bias-free walk-forward backtesting.

## Language

**Engine**:
A self-contained analysis pipeline that produces a full output block (regime classification, transition matrix, forecasts, walk-forward backtest) from a single methodology. Three engines exist: `threshold` (fast, close-only), `messina` (HMM with 19 Messina features), `hmm` (HMM with ~50 generic features). The user selects one per invocation.
_Avoid_: model, mode, method, strategy

**Walk-forward backtest**:
A bias-free backtest where at each bar `t`, only data up to `t-1` is used for regime classification. The position at bar `t` is determined by the regime at `t-1`. Produces discrete trade-level results: Sharpe, max drawdown, trade count, win rate, profit factor, total return.
_Avoid_: rolling backtest, historical simulation

**Regime**:
The market condition at a point in time, one of {Bear (0), Sideways (1), Bull (2)}. Bears are declining markets, Bullls are rising markets, Sideways are range-bound. State 0 always maps to Short, 1 to Flat, 2 to Long in the trading model.
_Avoid_: state (ambiguous — could mean HMM latent state or mapped regime)

**HMM latent state**:
The raw state index (0, 1, 2) output by a GaussianHMM `predict()` call. These indices are arbitrary and must be mapped to regimes by sorting ascending mean return: lowest → Bear, middle → Sideways, highest → Bull.
_Avoid_: state, cluster

**Signal**:
`P(next_regime = Bull) - P(next_regime = Bear)`, range [-1, 1]. Positive = bullish, negative = bearish. Derived from the transition matrix row for the current regime.
_Avoid_: score, conviction

**Transition matrix**:
A 3×3 row-normalized matrix where `T[i][j]` = probability of transitioning from regime `i` to regime `j`. Row sums to 1.0. Persistence diagonal entries (i → i) measure regime stickiness.
_Avoid_: confusion matrix, Markov chain

**OHLCV**:
Open, High, Low, Close, Volume — the five price columns required for feature engineering. Messina and HMM engines require OHLCV. The Messina and HMM engines require OHLCV and raise ``ValueError`` when missing.
_Avoid_: price data, market data

**Feature engineering**:
The process of computing technical indicators (log returns, SMAs, ATR, RSI, MACD, Bollinger, VSTOP, etc.) from OHLCV data. Two modes exist: `generic` (~50 indicators, SMA-based) and `messina` (19 indicators, Wilder's smoothing, VSTOP, ADX/DI, interaction terms, level gaps, KDJ oscillator). Features are precomputed once on the full dataset and sliced per bar in the walk-forward loop — all indicators are backward-looking (no lookahead bias).
_Avoid_: indicator calculation, TA computation

**Discrete trade**:
A trade with an entry time, entry price, exit time, exit price, and P&L. Positions are {-1, 0, 1}. A trade is opened when the regime changes and closed when it changes again. Transaction costs (commission, slippage) are modeled but default to zero for engine comparability.
_Avoid_: continuous position, fractional trade, signal-weighted trade

**Regime spooling**:
The threshold engine's method for mapping classified regimes to trading positions. At each bar, the regime (0/1/2) from `classify_regimes()` is mapped directly to a position via `{0: -1, 1: 0, 2: 1}`. No signal threshold or intermediate computation.
_Avoid_: signal-gating, conviction filtering

**BIC state selection** (`--n-states auto`):
Automatic selection of the number of HMM latent states via Bayesian Information Criterion. `select_n_states()` in `_hmm_shared.py` fits GaussianHMM for each candidate `k` in `[2, max_states]` with multiple restarts and returns the count with the lowest BIC. Default remains `--n-states 3`. The BIC penalty naturally guards against overfitting on short data windows.
_Avoid_: auto-tuning, state optimization

**PCA whitening**:
Optional dimensionality reduction applied after z-score normalization and before `model.fit()` inside `_fit_hmm_on_slice()`. Opt-in per engine via `pca_variance` parameter (e.g. 0.95 = keep 95% of variance). Reduces the generic engine's ~50 features to ~10-15 components. Component count is sticky — determined on first refit, reused on subsequent refits. Per ADR-0005.
_Avoid_: feature reduction, dimensionality reduction (use PCA whitening)

**Dwell-time filter** (`--dwell-bars N`):
Walk-forward filter that requires a regime to persist for N consecutive bars before the position changes. Lives in `walk_forward.py`, not inside engines (ADR-0003). Counter resets when the candidate regime changes. Default 0 = disabled.
_Avoid_: hold period, minimum hold

**Hysteresis filter** (`--hysteresis D`):
Walk-forward filter that only switches regimes when the posterior probability of the new regime exceeds the current regime's by a margin `> D`. Requires `posteriors` from `ClassifyResult` — no-op for the threshold engine (which returns `posteriors=None`). Lives in `walk_forward.py` (ADR-0003, ADR-0007). Default 0.0 = disabled.
_Avoid_: probability threshold, confidence filter

**Posteriors**:
Posterior probability array over regimes, returned by HMM engines via `ClassifyResult.posteriors`. Computed from `model.predict_proba()`. When `n_states > 3`, aggregated by regime bucket (Bear/Sideways/Bull). Used by the hysteresis filter. `None` for the threshold engine.
_Avoid_: probabilities, confidence scores

## Flagged ambiguities

- **"state"**: Use **regime** for the labeled market condition (Bear/Sideways/Bull) and **HMM latent state** for the raw model output index. Never use "state" alone.
- **"method"**: Use **engine** for the selected analysis pipeline. The old term "method" appears in the JSON `params.method` field; this is legacy from when threshold was the only engine and should migrate to `engine_info.method`.

## Example dialogue

> **Dev**: When the user runs `--engine hmm`, does the walk-forward backtest use the HMM's transition matrix or the threshold's?
>
> **Domain expert**: The HMM engine computes everything from the HMM. Each engine is self-contained. The HMM engine fits an HMM on the expanding feature window, predicts the regime at `t-1`, and maps that regime to a trade position. The transition matrix in the output comes from the HMM model parameters, not from threshold regime counts.
>
> **Dev**: And the signal field — is that still `P(bull) - P(bear)`?
>
> **Domain expert**: Yes, same formula, same range [-1, 1]. But the probabilities come from the HMM's transition matrix, not the threshold's. The formula is engine-agnostic; the source matrix differs.
>
> **Dev**: What happens if someone runs `--engine messina` on a CSV that only has close prices?
>
> **Domain expert**: It errors. Messina and HMM engines require OHLCV. The threshold engine is the only one that works with close-only data. If a caller passes a Series for engine=messina, `hmm_adapter.py` can synthetic-upgrade it (all columns = close), but that produces degenerate features.
