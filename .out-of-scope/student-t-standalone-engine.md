# Student-t Emissions as Standalone Engine

This project does not implement Student-t emissions as a standalone engine. Student-t emissions are being implemented as a shared utility in `_hmm_shared.py`, available to existing engines via a flag.

## Why a standalone engine is out of scope

Student-t emissions provide heavy-tailed alternatives to Gaussian emissions, with configurable degrees of freedom (ν) that degrade to Gaussian as ν→∞. The Student-t model is well-understood and the implementation is tractable.

However, implementing it as a standalone engine creates a maintenance problem:

- GH-HMM was proposed as a strict superset of Student-t (GH with λ=0 reduces to Student-t). Building both means maintaining two engines where one is mathematically redundant with the other.
- Even with GH-HMM deprioritized, Student-t as a *standalone* engine duplicates all the feature engineering, BIC selection, PCA whitening, and walk-forward integration already in `hmm` and `messina` engines. The only difference is the emission distribution.
- The right architecture is a shared utility in `_hmm_shared.py` (alongside `robust_fit_gaussian_hmm()`) that existing engines opt into via a flag. This follows the pattern established by the robust M-step utility.

## What we're doing instead

Student-t emission support will be added as a shared utility in `_hmm_shared.py`, available to `hmm` and `messina` engines via a `--student-t` CLI flag. This adds the tail-robustness benefit without engine proliferation.

## Prior requests

- #26 — "PRD: Student-t Emissions Engine — Heavy-Tailed HMM for Fat-Tailed Returns"

## Decision record

Closed 2026-05-29 after critical complexity review of issues #20–#28. Student-t retained as shared utility, not standalone engine. Full analysis posted on #20.
