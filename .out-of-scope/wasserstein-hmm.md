# Wasserstein HMM Engine

This project does not implement a Wasserstein-distance-based engine for regime identity tracking across walk-forward refits.

## Why this is out of scope

The Wasserstein engine proposed using 2-Wasserstein distance (W₂) between Gaussian emission distributions for template-based label stability across refits, replacing the existing Euclidean `_match_states()`.

The project uses `covariance_type="diag"` in all HMM engines. With diagonal covariances, the Wasserstein distance between two Gaussians collapses to:

```
W₂² = ‖μ₁−μ₂‖² + Σᵢ(σ₁ᵢ − σ₂ᵢ)²
```

This is barely more expressive than the existing Euclidean distance on means alone. The entire theoretical appeal of Wasserstein (matching full covariance structures) is neutered by the diag constraint.

Additionally:
- No evidence exists that label switching via `_match_states()` is a *demonstrated* user problem. The existing approach + BIC auto-selection + dwell/hysteresis filters adequately addresses label instability.
- The Boukardagha (2026) Sharpe 2.18 claim is on cross-asset daily data, not single-instrument futures, and has not been independently replicated.
- Template expiration as "adaptive complexity" is untested vs. BIC, which already provides data-driven state count selection.

## Prior requests

- #21 — "PRD: Wasserstein HMM Engine — Template-Based Label Stability"

## Decision record

Closed 2026-05-29 after critical complexity review of issues #20–#28. Full analysis posted on #20.
