# HDP-HMM Engine (Nonparametric Regime Discovery)

This project does not implement a Hierarchical Dirichlet Process HMM for nonparametric state count discovery.

## Why this is out of scope

The HDP-HMM proposed eliminating the `--n-states` parameter entirely by placing a Dirichlet process prior over the transition matrix, allowing the model to learn the effective number of states from data. The "sticky" variant adds a self-transition bias (κ) for persistent financial regimes.

This was rejected for three reasons:

1. **BIC already solves the problem.** The project already has `--n-states auto` via `select_n_states()` in `_hmm_shared.py`, which fits candidate state counts with multiple restarts and returns the BIC-optimal count. BIC is well-understood, computationally cheap, and already integrated. HDP-HMM replaces one automatic method with a more fragile one.

2. **New hyperparameters replace the one it eliminates.** The sticky HDP-HMM introduces α (DP concentration), κ (self-transition boost), and the Gibbs sampling iteration count. The κ parameter is particularly sensitive for financial data and requires tuning — you've traded `n_states` for three new knobs.

3. **Implementation fragility.** Gibbs sampling for HDP-HMM is non-trivial. The claim of "~200 lines" for a correct numpy implementation is optimistic. Correct Gibbs samplers for sticky HDP-HMM require careful handling of the Chinese restaurant process, beta distribution conjugacy, and convergence diagnostics. 500–1000 MCMC iterations per fit × refit-per-bar walk-forward changes the runtime profile from milliseconds to seconds per bar.

Additionally, the state-to-regime mapping ("when >3 states discovered, map multiple to same bucket") is underspecified and represents the core regime detection challenge left as an implementation detail.

## Prior requests

- #27 — "PRD: HDP-HMM Engine — Nonparametric Regime Discovery (No State Count Required)"

## Decision record

Closed 2026-05-29 after critical complexity review of issues #20–#28. Full analysis posted on #20.
