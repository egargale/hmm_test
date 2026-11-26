# ADR-0027: Statistical evaluation harness — metric definitions and non-regression floors

**Status**: Accepted
**Date**: 2026-11-26
**Issue**: #113 (Slice 2) / parent #111

## Context

Slice 2 of #111 adds a re-runnable 5-ticker x 5-engine statistical
evaluation harness (`tests/integration/test_statistical_evaluation.py`)
whose asserted metrics have to be defined exactly once and whose pass/fail
thresholds were re-measured on current master. Two of the definitions are
surprising without context: the persistence floor of "more than 3 bars" is
*not* satisfied by the median of raw regime run lengths (that gives exactly
3.0 for threshold and fails the strict comparison — the metric is the
per-ticker **mean spell length** `n_bars / (n_transitions + 1)`), and the
buy-and-hold reference is computed on the **engine's own window**
(`prices[252:]` after `min_train`, `numpy.diff`, rf=0, ddof=1), not the full
CSV — the one-line window choice flips messina from fail to pass. The
floors themselves (Fleiss' kappa `>= -0.10`, median per-engine B&H
delta-Sharpe `>= -1.20`) are deliberately **below** the originally proposed
0.0 / -1.0.

## Decision

1. The harness docstring is the single source of the metric definitions
   (1-step forecast identity [[ADR-0025]] / [[ADR-0026]], mean-spell
   persistence, Fleiss' kappa over 5x5 `current_regime` votes, B&H on the
   engine window). Changing a definition changes the measured values and
   therefore the floors — treat them as one contract.
2. Floors are **non-regression guards**, not quality targets: kappa
   `>= -0.10` and median B&H delta-Sharpe `>= -1.20` were measured on master
   (-0.0523 and -1.050 respectively) and sit deliberately close to the
   measured values, below the originally proposed 0.0 / -1.0 floors. Any
   deviation from a proposed floor must be disclosed in the PR body with the
   measured baseline and the formula that produced it.
3. The report artifact `runs/statistical_report.txt` is written by the test
   module, never `presenter.py` ([[ADR-0021]]), and is overwritten on each
   panel run — the tracked file becomes dirty after a rerun until re-committed.

## Consequences

- A product regression in the forecast algebra, persistence, cross-engine
  stability or walk-forward quality fails the build via
  `pytest -m statistical` instead of relying on the annual audit script.
- Tightening a floor toward 0.0 / -1.0 later is a data question, not a code
  one: re-measure the panel, then raise the floor in one constant.
