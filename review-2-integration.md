# Review 2: Protocol Compliance & Integration — PR #30 (af08b7f)

## Correct

### 1. RegimeEngine protocol — full compliance ✓
- **`robust_hmm.py`** `RobustHMMEngine` implements both required methods with correct signatures:
  - `precompute(self, data: pd.DataFrame) -> pd.DataFrame | None` (line 22)
  - `classify(self, data: pd.DataFrame, prev_means: np.ndarray | None = None) -> ClassifyResult` (line 25)
- `classify` returns `ClassifyResult` with all three fields populated: `regime` (int), `means` (np.ndarray, shape matching n_states), `posteriors` (np.ndarray). Verified at runtime.
- `@runtime_checkable` protocol: `isinstance(RobustHMMEngine(), RegimeEngine)` returns `True`. (test: `test_robust_hmm.py:52`)
- Consistent with existing engine pattern — none explicitly inherit from `RegimeEngine`; all use duck-typing via `@runtime_checkable`.

### 2. Engine registry — correctly wired ✓
- `engine_protocol.py:31` — `_build_registry()` includes `"robust_hmm": RobustHMMEngine`.
- `pipeline.py:28` — `_HMM_ENGINES = frozenset({"messina", "hmm", "robust_hmm"})`.
- `walk_forward.py:28` — `_HMM_ENGINES = frozenset({"messina", "hmm", "robust_hmm"})`.
- `walk_forward.py:27` — `_VALID_ENGINES = frozenset(ENGINE_REGISTRY.keys())` dynamically includes `robust_hmm` at import time because `_LazyRegistry` builds on first key access. Verified at runtime.
- `_ENGINE_FEATURES` in pipeline.py:39 includes `"robust_hmm": "generic"`.

### 3. Walk-forward integration — robust_method threaded correctly ✓
- `walk_forward.py:84` — `_resolve_engine(..., robust_method: str = "huber")` accepts and forwards the parameter.
- `walk_forward.py:102-103` — When `engine == "robust_hmm"`, `robust_method` is included in constructor kwargs.
- `walk_forward.py:118` — `walk_forward_backtest()` signature includes `robust_method: str = "huber"`.
- `walk_forward.py:153` — passes `robust_method` to `_resolve_engine()`.
- Verified at runtime: `_resolve_engine("robust_hmm", ..., robust_method="mcd")` creates engine with `engine.robust_method == "mcd"`.
- `robust_hmm` requires OHLCV (enforced via `_HMM_ENGINES` membership in `_resolve_engine` lines 92-96).

### 4. Pipeline integration — complete ✓
- `pipeline.py:72` — `run()` accepts `robust_method: str = "huber"`.
- `pipeline.py:144-145` — Passes `robust_method` to engine constructor when `engine == "robust_hmm"`.
- `pipeline.py:209` — Passes `robust_method` to `walk_forward_backtest()`.
- `pipeline.py:229` — `engine_info["robust_method"]` set when `engine == "robust_hmm"`.
- Temp engine instantiation for precompute (`pipeline.py:134`, `n_states=3`) doesn't need robust_method because `precompute()` only calls `engineer_features(data, use_messina=False)` — parameter independent of n_states and robust_method. Correct.

### 5. CLI wiring — plumbed through ✓
- `cli.py:192-198` — `--robust-method` with `choices=["huber", "mcd"]`, `default="huber"`.
- `cli.py:222` — `robust_method=args.robust_method` passed to `pipeline_run()`.
- `cli.py:153` — Engine `choices` updated to include `"robust_hmm"`.
- `--robust-method` is silently ignored for non-robust_hmm engines (pattern consistent with `--n-states` and `--pca-variance` for threshold). Help text is engine-specific: "Robust estimation method for robust_hmm engine".

### 6. csv_auto_detect.py — `_try_load_ohlcv()` correct ✓
- `csv_auto_detect.py:139-172` — Returns `None` when any OHLCV column is missing.
- Verified edge cases at runtime:
  - Close-only CSV → returns `None`
  - Full OHLCV CSV → returns DataFrame with columns `[open, high, low, close, volume]` and `DatetimeIndex`
  - Partial columns (open+high but no low) → returns `None`
- `load_prices` (line 189) now calls `_try_load_ohlcv(csv)` and routes OHLCV to all engines that need it. This is a feature improvement that makes `--engine messina` and `--engine hmm` work with CSVs that have OHLCV columns.
- Test updates in `test_regime_contract.py:113-121` properly test the error path via a synthetic close-only CSV in `tmp_path`.
- Test updates in `test_messina_integration.py:10-27` now assert success with BTC.csv (which has OHLCV). This is the correct new behavior.

### 7. No regressions ✓
- All 8 protocol & registry tests pass (`test_regime_engine.py::TestProtocolConformance` + `TestEngineRegistry`).
- All 4 `test_robust_hmm.py` protocol/registry tests pass.
- Existing engine test classes (ThresholdEngine, HMMGenericEngine, HMMMMessinaEngine) are unchanged.
- Test contract (`test_regime_contract.py`) still covers the OHLCV-error path with a close-only CSV fixture.
- `yfinance>=1.2.0` added as explicit dependency (`pyproject.toml:21`) — dependency hygiene fix for an already-used import.

## Blocker

None.

## Note

### N1: `robust_method` silently ignored for invalid values
**`_hmm_shared.py:175-179`** — `robust_fit_gaussian_hmm()` skips correction when `robust_method` is neither `"huber"` nor `"mcd"`, with no warning/error. The CLI restricts to valid choices (`choices=["huber", "mcd"]` at `cli.py:196`), so this can't be reached via CLI. However, programmatic callers could pass an invalid string and get silently uncorrected results.  
**Severity**: Low. Add an `else: raise ValueError(...)` in `robust_fit_gaussian_hmm` if guarding programmatic API misuse is desired.

### N2: MCD subsampling uses hardcoded seed
**`_hmm_shared.py:137`** — `rng = np.random.RandomState(0)` for MCD subsampling. This is independent of the HMM's `random_state=42`. Results are deterministic across calls but could mask issues if seed=0 produces a particularly lucky/unlucky subsample.  
**Severity**: Low. Consider parameterizing to match the HMM random_state for consistency.

### N3: `_try_load_ohlcv` re-parses the CSV
**`csv_auto_detect.py:142`** — `pd.read_csv(path)` is called a second time (first is in `load_from_csv`). For large files, this doubles parse time.  
**Severity**: Low. Could be optimized by passing the already-loaded DataFrame from `load_from_csv`, but the current approach keeps the function self-contained.

### N4: `_try_load_ohlcv` fallback date column
**`csv_auto_detect.py:162`** — If no date column is detected, falls back to `df.columns[0]`. If that column is numeric (e.g., row index), `pd.to_datetime()` produces epoch dates.  
**Severity**: Low. `_find_date_column` is robust and this only triggers with unusually formatted CSVs.
