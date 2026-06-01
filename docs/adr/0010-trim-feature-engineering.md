# ADR-0010: Trim Feature Engineering Monolith

**Status**: Accepted
**Date**: 2026-05-26
**Priority**: Worth Exploring
**Implemented**: 2026-05-27 (commit `e47ebf5`, PR #14 — FeatureEngineer class deleted, file 1300→849 lines)

## Context

`feature_engineering.py` is the largest file in the package at ~1,300 lines. It contains a top-level `add_features()` function (used by the active pipeline) plus an unused `FeatureEngineer` class with ~400 lines of feature selection, quality assessment, and summary methods. The class is dead code — the pipeline never instantiates it.

Meanwhile, `messina_features.py` shows the ideal shape: 237 lines, one focused function (`add_messina_features()`), 18 well-named features, no unused abstractions.

### What's Used vs Not

| In file | Used by active pipeline |
|---------|------------------------|
| `add_features()` | Yes — called by `hmm_adapter._engineer_features()` for generic HMM engine |
| 13× `_add_*()` helper functions | Yes — called by `add_features()` |
| `FeatureEngineer` class | No — never instantiated anywhere |
| `_downcast_float_columns()` | Yes — called by `add_features()` |
| `FeatureEngineer.add_features()` | No — wraps the top-level function, unused |
| `FeatureEngineer.apply_feature_selection()` | No — imports from `feature_selection.py` which is also dead |
| `FeatureEngineer.assess_feature_quality()` | No |
| `FeatureEngineer.get_enhanced_feature_summary()` | No |
| `FeatureEngineer.get_feature_importance()` | No |
| `FeatureEngineer.validate_features()` | No |

## Proposed Changes

### Step 1: Delete `FeatureEngineer` class (~400 lines)

Remove the entire class and its methods. This also eliminates the import chain through `feature_selection.py`.

### Step 2 (optional): Audit which `_add_*()` helpers produce features the HMM actually selects

`hmm_adapter._engineer_features()` for the generic engine does:
```python
df = add_features(data, min_periods=10)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
ohlcv_set = {"open", "high", "low", "close", "volume"}
cols = [c for c in numeric_cols if c not in ohlcv_set]
```

It takes **all** numeric features. Some `_add_*()` helpers may produce features that add noise without helping the HMM. An audit could identify which feature categories the HMM actually uses productively, and trim the rest.

### Step 3 (optional): Extract helpers into focused sub-modules

If the file still feels large after removing the class, group the `_add_*()` functions into sub-modules:
- `generic_features/volatility.py`
- `generic_features/momentum.py`
- `generic_features/trend.py`
- etc.

This mirrors the structure `messina_features.py` already has (single module, all features in one place).

## Consequences

**Positive:**
- File drops from ~1,300 to ~500 lines (active code only)
- No unused abstractions to confuse AI navigation
- `messina_features.py` and trimmed generic features become peers

**Negative:**
- If someone was using `FeatureEngineer` directly, their code breaks (unlikely — just renamed)
- Feature audit requires re-running HMM with reduced feature sets to measure impact

## Related

- [[ADR-0008]] Excise dead weight — `FeatureEngineer` removal overlaps with this PR; `feature_selection.py` deletion belongs in ADR-0008
- [[ADR-0009]] Engine seam — generic feature module would sit behind `HmmGenericEngine.precompute()`
