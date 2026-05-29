# GH-HMM Engine (Generalized Hyperbolic Emissions)

This project does not implement a Generalized Hyperbolic (GH) emission HMM engine.

## Why this is out of scope

The GH-HMM engine proposed using GH-distribution emissions (capturing skewness and heavy tails) with L1-penalized sparse precision matrices per regime. The GH family subsumes Gaussian, Student-t, and skewed distributions.

This was rejected for three reasons:

1. **Overparameterization.** GH emissions require 4 parameters per state (μ, Σ, λ, ψ) vs. 2 for Gaussian. With `n_states=3` and ~50 features, that's 600 tail/skew parameters — massive overparameterization risk for typical dataset sizes (252–2000 bars).

2. **diag covariance constraint.** The project uses `covariance_type="diag"`. The "sparse precision matrix" benefit (GraphLasso) is a no-op with diagonal covariances — the precision of a diagonal matrix is its reciprocal diagonal, already "sparse" by construction. The Foroni et al. (2024) paper validated with full covariance matrices.

3. **Numerical instability.** GH density evaluation requires `scipy.special.kv` (modified Bessel function of the second kind), which is numerically unstable at extreme parameter values — exactly what financial tail events produce.

Student-t emissions (capturing fat tails without skewness) provide 90% of the practical benefit at 20% of the implementation complexity. Student-t is being implemented as a shared utility in `_hmm_shared.py` instead.

## Prior requests

- #23 — "PRD: GH-HMM Engine — Generalized Hyperbolic Emissions + Sparse Precision Matrices"

## Decision record

Closed 2026-05-29 after critical complexity review of issues #20–#28. Full analysis posted on #20.
