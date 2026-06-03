# Degenerate-fit detection and recovery strategy

Date: 2026-06-02

## Status

Proposed

## Context

Live evaluation of all five engines (threshold, hmm, messina, robust_hmm, fshmm) against two tickers (0700.HK, CRM) over 10 years of daily data revealed three failure modes where HMM engines produce degenerate or misleading regime fits. A fourth issue — a 25× slowdown in walk-forward classification — was discovered in the same evaluation run.

This ADR codifies what constitutes a degenerate fit, how to detect it, and what recovery strategies are acceptable. It is the prerequisite for implementation work in [[ADR-#83]].

### Evidence source

CSV-based evaluation run, 2026-06-02. Data and results in `test_data/hmm-eval-2026-06-02/`. Two tickers (0700.HK = Tencent, CRM = Salesforce), five engines each, 2514 bars (CRM) and 2461 bars (0700.HK), daily frequency, 2016–06–01 to 2026–06–01.

### Walk-forward summary

| Ticker | Engine | Sharpe | Return | Trades | Persistence diag (bear/side/bull) | Degenerate? |
|--------|--------|--------|--------|--------|-----------------------------------|-------------|
| 0700.HK | threshold | +0.06 | −8% | 193 | 0.82/0.80/0.87 | No |
| 0700.HK | hmm | +0.02 | +10% | 5 | 0.98/1.00/0.99 | No |
| 0700.HK | messina | +0.06 | +5% | 11 | 0.97/0.99/0.98 | No |
| 0700.HK | robust_hmm | −0.05 | +2% | 7 | 0.97/1.00/0.97 | No |
| 0700.HK | **fshmm** | **+0.58** | **+268%** | 20 | 0.97/0.98/0.98 | No |
| CRM | threshold | −0.15 | −43% | 188 | 0.87/0.81/0.83 | No |
| CRM | hmm | −0.29 | −48% | 3 | 1.00/1.00/**0.95** | **Yes** — bull=0.8% |
| CRM | messina | −0.23 | −36% | 15 | 0.98/0.99/**0.95** | **Yes** — bull=2.4% |
| CRM | robust_hmm | −0.26 | −45% | 5 | 0.99/1.00/**0.95** | **Yes** — bull=1.6% |
| CRM | fshmm | −0.05 | −35% | 17 | 0.98/0.98/0.98 | No |

## Failure modes

### Mode 1: State collapse

**Engines affected**: hmm, messina, robust_hmm (all GaussianHMM-based).

**Symptom**: One or more states are assigned < 5% of total bars. Persistence diagonals for minority states approach an apparent "floor" (~0.95). Walk-forward produces very few trades (3–5 over 10 years), all losers (0% win rate).

**Root cause**: The GaussianHMM EM algorithm, when fit on high-dimensional feature spaces (~50 features for generic, ~19 for messina) with limited data (~2500 bars), overfits the emission distributions. The model discovers that concentrating almost all observations into 1–2 states minimizes the negative log-likelihood. The minority state still "exists" (the transition matrix has three rows) but is visited so rarely that it's effectively a 1–2 regime model.

**Note on the 0.95 "floor"**: Inspection of hmmlearn's M-step (`BaseHMM._do_mstep`) shows `transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)` followed by `normalize()`. With the default `transmat_prior=1.0`, there is no explicit floor — the 0.95 values emerge naturally when a state has very few observations and the posterior transitions are dominated by self-transitions in the E-step. The values like `0.9499999998149999` are numerical artifacts of the normalize step on a row with one dominant and two near-zero entries.

**Quantitative thresholds**:
- **State fraction**: Any state with < 5% of total bars → degenerate
- **Trade floor**: Walk-forward produces < 10 trades over 2500+ bars → suspect (may be caused by degenerate fit or legitimately stable data)
- **Persistence ceiling**: Any diagonal > 0.99 → warning (extremely sticky transitions reduce model usefulness)

**Ticker dependency**: On 0700.HK, all HMM engines produce balanced state distributions (no state below 7%). On CRM, three of four HMM engines collapse. This is not an engine bug — it's a data × feature-space mismatch. CRM's return distribution ( Salesforce, US tech) is less regime-structured than 0700.HK (HSBC-listed, Chinese regulatory cycles create clearer regime breaks).

### Mode 2: Feature saliency instability

**Engine affected**: fshmm.

**Symptom**: Feature saliency weights are unstable across walk-forward refits, producing inconsistent regime assignments. In extreme cases (old eval), the model inverts regime polarity — labeling an uptrend as bear. In the fresh eval, fshmm/CRM produces near-zero signal (+0.0011) with mediocre results (−35% return), while fshmm/0700.HK is the best performer across all engines (+268%).

**Root cause**: The FS-HMM algorithm estimates per-feature saliency weights (ρ) alongside the HMM parameters. With ~50 features and ~2500 bars, the saliency estimation has insufficient data to distinguish signal features from noise. On 0700.HK, the 2021 regulatory crash provides a strong regime signal that stabilizes saliency. On CRM, no such clean break exists.

**Quantitative threshold**:
- **Data sufficiency**: `n_bars < 4 × n_features × n_states` → low-data warning
- For CRM: 2514 < 4 × 50 × 3 = 600 → technically sufficient, but saliency stability should be checked
- **Saliency variance**: If the top-5 feature ranks change by > 50% across walk-forward steps, saliency is unstable

### Mode 3: Over-robustness

**Engine affected**: robust_hmm.

**Symptom**: The Huber/MCD robust estimator absorbs genuine regime changes as "outlier noise", producing regime counts nearly identical to standard hmm but with worse walk-forward performance.

**Evidence**:

| Ticker | Engine | Bear% | Side% | Bull% | Sharpe | Return |
|--------|--------|-------|-------|-------|--------|--------|
| 0700.HK | hmm | 8.1% | 81.0% | 10.9% | +0.02 | +10% |
| 0700.HK | robust_hmm | 7.3% | 85.0% | 7.6% | −0.05 | +2% |
| CRM | hmm | 43.0% | 56.2% | 0.8% | −0.29 | −48% |
| CRM | robust_hmm | 42.2% | 56.2% | 1.6% | −0.26 | −45% |

CRM robust_hmm ≈ CRM hmm (regime counts differ by < 1%). The robust estimator adds no value but costs ~10% more wall time.

**Quantitative threshold**:
- **Over-robustness ratio**: If robust_hmm regime counts differ from hmm by < 10% on any state, the robust correction is not differentiating.
- **Cost-benefit**: robust_hmm should improve Sharpe by > 0.05 over hmm to justify its computational overhead. Currently it's worse on both tickers.

### Walk-forward timing reference (serial execution)

Investigation of a suspected 25× slowdown ([[Issue #86]], closed as not-a-bug) established these baseline timings on 2514-bar CRM data:

| Engine | Wall (s) | Median/step (s) | Relative |
|--------|----------|-----------------|----------|
| threshold | 1.4 | — | 1× (baseline) |
| messina | 23 | 0.17 | 16× | 
| hmm | 38 | 0.33 | 27× |
| robust_hmm | 95 | 0.40 | 68× (Huber IRLS overhead) |
| fshmm | 135 | 1.13 | 96× (saliency EM overhead) |

These are serial timings. Concurrent execution inflates wall time due to CPU contention on EM fitting. Eval harness (#82) should run serially for accurate timing.

## Decision

### Detection: warn-and-proceed (do not auto-retry)

When a degenerate fit is detected, the engine will:

1. **Add diagnostic fields** to `engine_info`:
   - `degenerate_fit: bool` — any state < 5% of bars
   - `degenerate_caveat: str` — human-readable description
   - `low_data_warning: bool` — insufficient data for the engine's feature space
   - `over_robustness: bool` — robust_hmm not differentiating from hmm

2. **Penalize confidence** in the verdict proportionally to the severity of the degenerate fit.

3. **Not auto-retry** with different parameters. Auto-retry introduces hidden complexity:
   - Which fallback parameters? (fewer states, PCA, different engine?)
   - How many retries before giving up?
   - Results become non-reproducible (same input → different output depending on retry path)
   - Users can't learn from failures if they're silently hidden

Instead, the detection output gives users (and downstream consumers like walk-forward) the information to make their own decision.

### Recovery: user-driven fallbacks

The ADR documents acceptable fallback strategies that users can invoke explicitly:

| Strategy | CLI flag | When to use |
|----------|----------|-------------|
| Reduce feature dimensionality | `--pca-variance 0.95` | State collapse on generic HMM engines |
| Reduce state count | `--n-states 2` | 3-state model collapses; data may be 2-regime |
| Auto-select state count | `--n-states auto` | Not sure if 2 or 3 regimes; let BIC decide |
| Switch engine | `--engine messina` | Generic features too noisy; messina's 19 hand-picked features are more stable |
| Switch engine | `--engine threshold` | HMM is overkill for this ticker; threshold's simple return-based approach is more robust |
| Add whipsaw filter | `--hysteresis 0.1` | Model is switching too often (not the current problem, but available) |

### Minimum data recommendations

| Engine | Min features | Min bars | Formula |
|--------|-------------|----------|---------|
| threshold | 1 | 60 | 20 bars per state × 3 states |
| messina | 19 | 228 | 4 × 19 × 3 |
| hmm | ~50 | 600 | 4 × 50 × 3 |
| robust_hmm | ~50 | 600 | Same as hmm (plus extra for robust estimation) |
| fshmm | ~50 | 600 | Same as hmm (saliency adds parameters but same feature count) |

These are minimums for 3-state models. With `--n-states auto` and BIC selection, the effective minimum may be lower since BIC may select 2 states.

### Timing expectations

Expected serial wall times for a ~2500-bar dataset (3 states, no PCA):

| Engine | Target | Timeout warning |
|--------|--------|----------------|
| threshold | < 5s | > 10s |
| hmm | < 60s | > 120s |
| messina | < 30s | > 60s |
| robust_hmm | < 120s | > 240s |
| fshmm | < 180s | > 360s |

These are per-engine serial limits. Parallel execution will exceed these proportionally.

## Considered options

### A) Auto-retry with fallback (rejected)

On detecting a degenerate fit, automatically retry with PCA, fewer states, or a different engine.

**Rejected because**: (1) Makes output non-reproducible — same input can produce different results depending on retry path. (2) Masks the underlying data problem from users. (3) Complex retry logic needs its own testing and may introduce new bugs. (4) The "right" fallback is context-dependent (are you trading or researching?).

### B) Hard error on degenerate fit (rejected)

Raise an exception and refuse to produce output when a degenerate fit is detected.

**Rejected because**: (1) A degenerate fit still contains useful information — the fact that the model can't find three regimes IS a signal about the data. (2) Breaks existing workflows and scripts. (3) The threshold for "degenerate" is a heuristic, not a binary truth — hard errors on heuristics are fragile.

### C) Warn-and-proceed with diagnostic fields (chosen)

Add diagnostic fields to the output, penalize confidence in the verdict, but always produce a result.

**Chosen because**: (1) Preserves backward compatibility. (2) Downstream consumers (walk-forward, UI, API) can check the diagnostic fields and decide. (3) Results are reproducible. (4) Users learn about data characteristics from the warnings. (5) Aligns with the project's philosophy of "give the user the tools to decide, don't decide for them".

## Consequences

### New fields in engine output

When a degenerate fit is detected, `engine_info` gains:

```python
{
    "method": "hmm",
    "features": "generic",
    "n_states": 3,
    "warmup_bars": 252,
    # New diagnostic fields (only present when triggered)
    "degenerate_fit": true,
    "degenerate_caveat": "bull state has 0.8% of bars (20/2514); model is effectively 2-regime",
    "low_data_warning": false,
    "over_robustness": false
}
```

When low data is detected:

```python
{
    "method": "fshmm",
    "features": "generic",
    "n_states": 3,
    "low_data_warning": true,
    "low_data_caveat": "2514 bars with 50 features × 3 states; recommend minimum 600 bars"
}
```

When robust_hmm is not differentiating:

```python
{
    "method": "robust_hmm",
    "features": "generic",
    "n_states": 3,
    "over_robustness": true,
    "over_robustness_caveat": "regime counts differ from hmm by <10% on all states; robust correction not adding value"
}
```

### Verdict confidence penalty

When `degenerate_fit` is true, the verdict confidence is scaled down:

```
adjusted_confidence = raw_confidence × min(state_fraction) / 0.05
```

This means if the smallest state has 0.8% of bars, confidence is multiplied by 0.8/5 = 0.16. A raw confidence of 0.98 becomes 0.16 — clearly telling the user "this model is uncertain".

### Cross-references

- [[ADR-0005]] PCA whitening — the recommended first fallback for state collapse
- [[ADR-0006]] BIC state count selection — `--n-states auto` as a recovery strategy
- [[ADR-0007]] Hysteresis and dwell-time filters — available but not the primary fix for state collapse
- [[ADR-0013]] FS-HMM engine — saliency instability is fshmm-specific
- [[ADR-0016]] Robust HMM engine — over-robustness is robust_hmm-specific

### No changes to regime assignments

This ADR covers **detection only**. The actual regime assignments (which bars are bear/sideways/bull) are not modified by degenerate-fit detection. Recovery strategies (PCA, fewer states, engine switch) are user-driven via CLI flags.

### Walk-forward interaction

The walk-forward backtest reads the diagnostic fields from each step's classify output. When a step reports `degenerate_fit: true`, the walk-forward may:
- Log a warning (always)
- Optionally skip the step's signal (configurable) — see [[ADR-#84]]
- Apply signal-scaled sizing to reduce exposure during uncertain steps — see [[ADR-#85]]

These downstream behaviors are defined in their respective issues, not in this ADR.

### Implementation

Implementation is tracked in [[Issue #83]] — this ADR provides the thresholds and strategy; #83 provides the code.
