# HMM Theory for Regime Detection

## Overview

Hidden Markov Models (HMMs) model a system where an unobserved (hidden) state sequence generates observable data. For market regime detection, the hidden states represent the underlying market condition (bear/sideways/bull), and the emissions are price-derived features (returns, volatility, etc.).

## Model Structure

An HMM is defined by three components:

### 1. Initial State Distribution (π)
A vector of length `n_states` (default 3). The probability of starting in each state. Learned from data during fitting.

### 2. Transition Matrix (A)
An `n_states × n_states` row-stochastic matrix where `A[i,j]` = probability of transitioning from state `i` to state `j`. The diagonal (`A[i,i]`) measures **persistence** — how "sticky" each regime is. A persistence of 0.90 means there is a 90% chance of staying in the same regime next period.

### 3. Emission Distribution (B)
The probability distribution of observed features given a state. This skill uses:

- **Gaussian HMM**: Each state emits features from a multivariate Gaussian distribution with its own mean vector `μ_i` and covariance matrix `Σ_i`. This is the default and works well for most financial time series.
- **GMM HMM**: Each state emits from a Gaussian Mixture Model (multiple Gaussians per state). Captures multi-modal distributions within a regime but requires more parameters and data.

## Training: The EM Algorithm (Baum-Welch)

HMMs are trained via the Expectation-Maximisation (EM) algorithm, also known as the Baum-Welch algorithm:

1. **Initialisation**: Randomly initialise transition matrix, emission parameters.
2. **E-step**: Compute expected state occupancy and state transition counts given current parameters (forward-backward algorithm).
3. **M-step**: Re-estimate transition matrix and emission parameters to maximise expected log-likelihood.
4. **Repeat** until convergence (`tol`) or max iterations (`n_iter`).

Convergence is sensitive to initialisation. The `random_state` parameter ensures reproducibility. Multiple restarts (`num_restarts`) with different seeds can improve fit for noisy financial data.

## State Inference: The Viterbi Algorithm

After training, the Viterbi algorithm finds the single most likely sequence of hidden states given the observed features:

```
argmax_{s_1...s_T} P(s_1...s_T | o_1...o_T)
```

This is used by `run_hmm_regime()` to label each bar with a regime. Viterbi uses dynamic programming (similar to Dijkstra) to find the optimal path through the state trellis.

## State Ordering Caveat

**HMM states are unlabeled during training.** State 0, 1, 2 have no inherent meaning — the model just partitions the feature space. The skill's `run_hmm_regime()` re-orders states post-hoc by **ascending mean return**:

1. Compute the mean of the log-return feature for each state.
2. Sort states by this mean: lowest → **Bear**, middle → **Sideways**, highest → **Bull**.

**This labeling can swap between re-fits.** If you re-train on different data or with different `random_state`, State 0 from run 1 might be Bull and State 0 from run 2 might be Bear. Agents must not rely on numeric state indices — always use the labeled "bear"/"sideways"/"bull" strings.

## Stationary Distribution

The stationary distribution `π*` satisfies `π* = π* × A`. It represents the long-run fraction of time spent in each regime if the transition dynamics hold indefinitely. Computed as the left eigenvector of `A` with eigenvalue 1.

A stationary distribution of `{bear: 0.30, sideways: 0.45, bull: 0.25}` means the market is expected to be in a bear regime 30% of the time in the long run.

## Number of States

The default `n_states=3` maps to the bear/sideways/bull framework. Two states are sometimes used for up/down classification. Four or more states can capture sub-regimes (e.g. "strong bull" vs "weak bull") but require more data and are harder to interpret.

AIC/BIC criteria can guide state count selection: fit models with 2–6 states and choose the one minimising the information criterion. The `_compute_information_criteria` method on `BaseHMMModel` provides this if needed.

## Limitations

1. **Markov assumption**: The next state depends only on the current state, not the full history. Financial markets may exhibit longer memory.
2. **Stationarity**: Transition matrix and emission parameters are assumed constant over time. In reality, regime dynamics evolve.
3. **Gaussian assumption**: Financial returns have fat tails — the Gaussian emission model may underestimate extreme moves.
4. **Data requirements**: HMMs with `n_states=3` and `covariance_type="full"` need at least several hundred observations. Fewer than ~100 bars → near-certain convergence failure.

## References

- Baum, L.E. & Petrie, T. (1966). "Statistical Inference for Probabilistic Functions of Finite State Markov Chains"
- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
- hmmlearn documentation: https://hmmlearn.readthedocs.io/
