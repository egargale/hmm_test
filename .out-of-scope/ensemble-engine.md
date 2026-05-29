# Ensemble Engine (Multi-Engine Voting)

This project does not implement a multi-engine voting/ensemble wrapper.

## Why this is out of scope

The ensemble engine proposed wrapping a configurable subset of engines from the `ENGINE_REGISTRY` and aggregating predictions via majority, average, or weighted-BIC voting strategies.

The ensemble's value proposition depends entirely on engine *diversity* — different failure modes that cancel out when combined. In practice:

- All existing and planned engines are HMM variants using the same `diag` covariance type, the same feature engineering pipeline, and the same walk-forward infrastructure. Their failure modes are correlated, not independent.
- With 3 engines (the common case), majority voting produces ties when each engine picks a different regime — not a corner case, but the expected behavior for Sideways regimes.
- "Weighted-by-BIC" assumes BIC is comparable across different model families, but BIC is not designed for cross-family comparison.
- Auto-discovery of engines from the registry silently changes ensemble behavior when new engines are added, with no user opt-in.

If the engine set becomes genuinely heterogeneous (e.g., adding a threshold-based engine alongside deep-learning or structural models), ensemble voting could be revisited. For the current HMM-only ecosystem, it adds architectural surface area without statistical benefit.

## Prior requests

- #25 — "PRD: Ensemble Engine — Multi-Engine Voting for Robust Regime Detection"

## Decision record

Closed 2026-05-29 after critical complexity review of issues #20–#28. Full analysis posted on #20.
