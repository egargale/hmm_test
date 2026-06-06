> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: git_detective.py — 90-day window (129 commits, 2 authors)

# Git Hotspots & Coupling Analysis

## Overview

- **Period**: 90 days | **Commits**: 129 | **Authors**: 2

## Hotspots

### 🔴 High Risk (> 15 changes)

| Path | Changes | Note |
|------|---------|------|
| `regime/pipeline.py` | **35** | Central orchestrator |
| `cli.py` | **25** | +1 from yfinance cache flags |
| `regime/engine_protocol.py` | **18** | Protocol + registry |
| `CONTEXT.md` | **15** | +1 from nexus-query section |

### 🟡 Medium Risk (5-15)

| Path | Changes |
|------|---------|
| `regime/walk_forward.py` | 14 |
| `regime/engines/hmm_generic.py` | 13 |
| `regime/engines/hmm_messina.py` | 13 |
| `SKILL.md` | 13 |
| `regime/engines/_hmm_shared.py` (ghost) | 12 |
| `regime/engines/fshmm.py` | 11 |
| `regime/engines/robust_hmm.py` | 11 |

### New files (1 commit — not yet in hotspots)

- `data_processing/ticker_cache.py` — 96 lines
- `tests/unit/test_ticker_cache.py` — 201 lines, 7 test classes
- `tests/unit/test_cli_cache_flags.py` — 136 lines

## Critical Coupling

| Score | Pair | Co-changes |
|-------|------|-----------|
| **1.00** | pipeline.py ↔ walk_forward.py | 14 |
| **1.00** | hmm_generic.py ↔ hmm_messina.py | 13 |
| **0.94** | pipeline.py ↔ engine_protocol.py | 17 |
| **0.90** | engine_protocol.py ↔ test_regime_engine.py | 9 |

## Implications

1. **pipeline.py** remains the highest-risk file (35 changes, 32 dependents).
2. **cli.py** crossed 24→25 changes (yfinance cache flags).
3. **ticker_cache.py** is only 1 commit old — too new for hotspot data.
4. **Ghost file** `_hmm_shared.py` (12 changes) referenced in git history but gone from the tree.
