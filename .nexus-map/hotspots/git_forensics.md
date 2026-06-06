> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: git_detective.py — 90-day analysis window (127 commits, 2 authors)

# Git Hotspots & Coupling Analysis

## Overview

- **Analysis period**: 90 days (2026-03-08 to 2026-06-06)
- **Total commits**: 127
- **Authors**: 2
- **Analysis command**: `git_detective.py --days 90`

## Hotspots (sorted by risk)

### 🔴 High Risk (> 15 changes)

| Path | Changes | Description |
|------|---------|-------------|
| `hmm_futures_analysis/regime/pipeline.py` | **35** | Central pipeline orchestrator |
| `hmm_futures_analysis/cli.py` | **24** | CLI entrypoint |
| `hmm_futures_analysis/regime/engine_protocol.py` | **18** | Engine protocol + registry |

### 🟡 Medium Risk (5-15 changes)

| Path | Changes | Description |
|------|---------|-------------|
| `CONTEXT.md` | 14 | Domain language documentation |
| `hmm_futures_analysis/regime/walk_forward.py` | 14 | Walk-forward backtest |
| `hmm_futures_analysis/regime/engines/hmm_generic.py` | 13 | Generic HMM engine |
| `hmm_futures_analysis/regime/engines/hmm_messina.py` | 13 | Messina HMM engine |
| `SKILL.md` | 13 | Agent skill definition |
| `hmm_futures_analysis/regime/engines/_hmm_shared.py` | 12 | *ghost — no longer exists* |
| `pyproject.toml` | 11 | Build config + dependencies |
| `hmm_futures_analysis/regime/engines/fshmm.py` | 11 | FSHMM engine |
| `hmm_futures_analysis/regime/engines/robust_hmm.py` | 11 | Robust HMM engine |
| `tests/integration/test_regime_engine.py` | 10 | Integration: engine contract |
| `hmm_futures_analysis/regime/engines/_hmm_pipeline.py` | 9 | HMM pipeline helpers |
| `tests/integration/test_regime_pipeline.py` | 9 | Integration: pipeline |
| `docs/adr/README.md` | 9 | ADR index |

## Critical Coupling Pairs

### Perfect co-change (1.00)
- **pipeline.py** ↔ **walk_forward.py** (14 co-changes)
  - These always change together despite being separate source files. Walk-forward is effectively an internal implementation detail of the pipeline.
- **hmm_generic.py** ↔ **hmm_messina.py** (13 co-changes)
  - Mirror engines evolving identically. Any change to the shared base class (`_hmm_engine.py`) or pipeline interface propagates to both.

### Very high coupling (> 0.80)
- **pipeline.py** ↔ **engine_protocol.py** (0.94, 17 co-changes)
  - Protocol changes always require pipeline integration updates.
- **fshmm.py** ↔ **robust_hmm.py** (0.82, 9 co-changes)
  - Newer HMM engines added together, sharing patterns.
- **fshmm.py** ↔ **pipeline.py** (0.82, 9 co-changes)
- **engine_protocol.py** ↔ **test_regime_engine.py** (0.90, 9 co-changes)

## Implications

1. **pipeline.py is the single highest-risk file** (35 changes). Any modification has a 32-module impact radius.
2. **The pipeline/walk_forward/engine_protocol triad is tightly coupled** — treat them as one unit for change planning.
3. **Engine mirroring** (hmm_generic ↔ hmm_messina at 1.00) suggests opportunities for further shared code consolidation or alternatively confirms the shared base class is working.
4. **Ghost file** `_hmm_shared.py` (12 changes) was deleted/renamed — raw data refers to a file that no longer exists in the working tree.
