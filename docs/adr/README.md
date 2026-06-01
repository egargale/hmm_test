# Architecture Decision Records

This directory documents all significant architecture decisions for the HMM futures trading analysis project.

## Numbering scheme

ADRs use a flat, sequential 4-digit numbering scheme (`0001-`, `0002-`, …).

| Number | File | Subject |
|--------|------|---------|
| 0001 | `0001-three-independent-engines.md` | Three independent regime-detection engines |
| 0002 | `0002-same-repo-dual-distribution.md` | Same repo, dual distribution |
| 0003 | `0003-engine-self-containment.md` | Engine self-containment contract |
| 0004 | `0004-cli-data-loading-seam.md` | CLI data loading seam |
| 0005 | `0005-pca-in-model-layer.md` | PCA placement in model layer |
| 0006 | `0006-bic-state-count-selection.md` | BIC-based HMM state count selection |
| 0007 | `0007-hysteresis-dwell-time-filters.md` | Hysteresis and dwell-time whipsaw filters |
| 0008 | `0008-excise-dead-weight.md` | Excise dead weight modules |
| 0009 | `0009-deepen-engine-seam.md` | Deepen engine seam with protocol |
| 0010 | `0010-trim-feature-engineering.md` | Trim feature engineering monolith |
| 0011 | `0011-engine-dispatch-consolidation.md` | Engine dispatch consolidation |
| 0012 | `0012-pipeline-run-decomposition.md` | Pipeline `run()` decomposition |
| 0013 | `0013-fshmm-engine.md` | Feature Saliency HMM (FSHMM) engine |
| 0014 | `0014-per-phase-timing-instrumentation.md` | Per-phase timing instrumentation |

## Rules for contributors

1. **Numbers are never reused.** Once assigned, an ADR number is permanent — even if the ADR is superseded or deprecated.
2. **New ADRs get the next available number** (e.g., `0013-` after the current highest).
3. **Cross-references** between ADRs use the `[[ADR-NNNN]]` format (4-digit). The wikilink convention matches the 4-digit scheme so that `[[ADR-0008]]` links to `0008-excise-dead-weight.md` in Obsidian and similar tools.
4. **Filename format:** `NNNN-kebab-case-title.md` where `NNNN` is the zero-padded 4-digit number.
5. **No two ADRs may share a number.** If this happens (e.g., during a merge), the conflict must be resolved before the PR is accepted.

## History

Originally, this project had two ADR directories:
- `docs/architecture/` — 3 ADRs using a 3-digit `ADR-001` scheme with a formal template
- `docs/adr/` — 7 ADRs using a 4-digit `0001-` scheme with free-form markdown

These were consolidated into a single `docs/adr/` directory with the 4-digit scheme. The original `docs/architecture/` ADRs became `0008`–`0010`.
