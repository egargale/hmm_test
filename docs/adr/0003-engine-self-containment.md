# Engine self-containment contract

Before issue #10 was fixed, the pipeline used threshold-based regime
classification for the top-level output block (transition matrix, stationary
distribution, persistence, signal, regime counts, forecasts) regardless of
which engine was selected. The `--engine` flag only affected the walk-forward
backtest. This produced identical top-level output across all three engines,
silently contradicting the self-contained design established in ADR-0001.

We need to codify the self-containment contract so that future changes cannot
regress the separation — either by accident (a contributor not realizing the
contract exists) or by convenience (reusing another engine's output to avoid
duplicating work).

## Considered options

### A) Implicit contract — rely on ADR-0001 and code review

ADR-0001 already states engines are self-contained. Trust that code review
and the existing documentation prevent regressions.

**Rejected because**: ADR-0001 is an architectural decision about having three
engines, not a contract about what each engine must own. The bug in issue #10
proved that "self-contained" was interpreted loosely — contributors treated
it as "the walk-forward backtest is engine-specific" while leaving the
top-level block shared. A general statement is not a substitute for an
explicit enumeration of what each engine must produce.

### B) Enforce via engine protocol (code-level contract)

Add assertions or tests that verify each engine's output differs from the
other engines when given the same input. Enforce self-containment at the
test suite level rather than documenting it.

**Rejected because**: testing that outputs differ is a necessary guard but
not sufficient — it catches regressions after they land, not before they're
written. Tests also can't express *why* the contract exists or what a
contributor should consider when adding a new engine. Code-level contracts
and documentation serve different purposes; this ADR is the documentation
layer.

### C) Explicit self-containment ADR (chosen)

Document the contract as a standalone ADR that enumerates every output field
each engine must own, states what is excluded, and explains the motivation.
Contributors adding new engines or modifying existing ones can read this ADR
and know exactly what boundaries to respect.

**Chosen because**: (1) a concrete enumeration of owned fields prevents the
vague "self-contained" interpretation that caused issue #10; (2) it serves
as a checklist when adding a new engine — the contributor knows exactly
which outputs they must produce; (3) it explains the *why* (prevent silent
regression), which tests alone cannot do; (4) it clarifies the walk-forward
backtest's separate-concern status, which was a source of confusion in the
original bug.

## Consequences

- **Each engine owns seven top-level output fields.** Regime sequence,
  transition matrix, stationary distribution, persistence diagonal, signal,
  regime counts, and forecasts must all derive from the selected engine's
  own regime classification. No field may silently reuse another engine's
  computation for the top-level output block.
- **The walk-forward backtest is a separate concern.** It already uses
  engine-specific regime paths and has its own output block
  (`walk_forward`). The self-containment contract applies to the top-level
  output, not to the backtest internals. No change needed.
- **New engines must produce all seven fields.** Adding a fourth engine
  means implementing regime classification that feeds into all seven outputs.
  A new engine cannot piggyback on an existing engine's transition matrix
  or signal computation.
- **Same output schema, different source data.** All engines produce the
  same JSON shape — consumers read `result["transition_matrix"]` regardless
  of engine. The *values* differ because the *source* differs, but the
  *structure* is identical. This is unchanged from ADR-0001.
- **Testing responsibility.** The `test_engine_independence` test suite
  verifies that different engines produce different top-level outputs for
  the same input. This ADR documents the contract; the tests enforce it.
  Both layers are needed.
- **CONTEXT.md example dialogue is consistent.** The existing dialogue
  already describes the post-fix behavior correctly — each engine computes
  its own transition matrix and signal. No update needed.
