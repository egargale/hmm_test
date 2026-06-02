# PRD: Best HMM Implementation for Stocks

## Goal
Transform the existing HMM codebase into a production-grade regime detection pipeline by closing the documented gaps between SOTA research and current implementation.

## Tasks

### Task 1: Config — Add new Pydantic configs
**File**: `src/utils/config.py`  
**Changes**: Add `BICConfig`, `WalkForwardConfig`, `HysteresisConfig` classes.
- `BICConfig`: min_states=2, max_states=7, restarts_per_state=50, max_iter=500, tol=1e-6
- `WalkForwardConfig`: train_days=1260(5yr), step_days=21, min_train_days=252
- `HysteresisConfig`: enter_threshold=0.6, exit_threshold=0.4, min_consecutive=3
- Add these as nested models in `Config` with defaults

### Task 2: BIC Selector — New module for data-driven state count
**File**: `src/hmm_models/bic_selector.py` (NEW)  
**Function**: `select_best_n_states(features, config) -> (best_n, bic_scores, best_model)`
- Sweep n_states from BICConfig.min_states to max_states
- For each n_states: run BICConfig.restarts_per_state random restarts, pick best log-likelihood
- Compute BIC = -2 * log_likelihood * n_samples + n_params * log(n_samples)
- n_params = (n_states-1) + n_states*(n_states-1) + n_states*n_features + n_states*n_features*(n_features+1)/2 (for "full" cov)
- Use joblib.Parallel for multi-restart parallelism
- Return best_n, list of (n_states, bic, model), best_model
- Must handle GaussianHMM training failures gracefully (skip failed restarts)

### Task 3: Walk-Forward — Expanding-window validation harness
**File**: `src/model_training/walk_forward.py` (NEW)  
**Function**: `run_walk_forward(df, features, config) -> dict with oos_predictions, oos_probs, models`
- For each fold: train on expanding window (index 0 to current_end)
- Predict on test window (current_end to current_end+step_days)
- Fit StandardScaler on training data only, transform test data with it
- Train new model each fold (with BIC selection if configured)
- Concatenate all out-of-sample predictions
- Return dict with: oos_states, oos_probabilities, models list, fold_info

### Task 4: Feature Engineering — Fix windows + add PCA
**File**: `src/data_processing/feature_engineering.py`  
**Changes**:
- Fix default indicator windows in `get_default_indicator_config()` to industry standards: RSI=14, ATR=14, BB=20, MACD=(12,26,9), Stochastic=(14,3), ADX=14
- Add `add_pca_whitening(features, variance_threshold=0.95)` function
- Function should: StandardScaler -> PCA -> return transformed features + PCA object
- Store PCA in config for model persistence

### Task 5: Gaussian HMM — BIC method + state relabeling
**File**: `src/hmm_models/gaussian_hmm.py`  
**Changes**:
- Add `select_n_states_bic(X, min_states=2, max_states=7, restarts=50)` method
- Add `relabel_states_by_return(X)` method that reorders states so state 0 = highest mean log-return
- Add `relabel_states_by_volatility(X)` method as alternative (state 0 = lowest vol)
- State relabeling must also reorder transmat_, means_, covars_ accordingly
- These should work with or without the BIC selector module

### Task 6: HMM Trainer — 50 restarts default
**File**: `src/model_training/hmm_trainer.py`  
**Changes**:
- Change default `num_restarts` from 3 to 50 in `HMMConfig`
- Add `train_with_bic_selection()` function that wraps BIC selector + training
- Update `train_model()` signature to accept `BICConfig` and do BIC selection if provided

### Task 7: Inference Engine — Online forward filter
**File**: `src/model_training/inference_engine.py`  
**Changes**:
- Add `OnlineHMMForwardFilter` class that implements the forward algorithm in real-time
- Constructor: `OnlineHMMForwardFilter(model, scaler)`
- `init_belief()`: set initial state distribution from model.startprob_
- `update(observation)`: one-step forward filter update
  - prior = belief @ transmat
  - likelihood = multivariate_normal.pdf(observation, means, covars)
  - posterior = prior * likelihood / sum(prior * likelihood)
  - belief = posterior
  - return posterior
- `observe(observations)`: batch forward filter (sequential update for each observation)
- `reset()`: reset belief to initial distribution

### Task 8: Strategy Engine — Hysteresis + dwell filters
**File**: `src/backtesting/strategy_engine.py`  
**Changes**:
- Add hysteresis parameters to `backtest_strategy()`: enter_threshold, exit_threshold, min_consecutive_periods
- Implement dwell filter: state change is only executed when P(regime) > enter_threshold
- Implement confirmation counter: require N consecutive periods above threshold before acting
- Exit when P(regime) drops below exit_threshold
- Maintain current position, only change when dwell conditions are met
- Add `HysteresisResult` dataclass for tracking dwell state

### Task 9: Best HMM Pipeline — Orchestrator
**File**: `src/pipelines/best_hmm_pipeline.py` (NEW)  
**Function**: `run_best_hmm_pipeline(df, config) -> dict of results`
- Orchestrates: feature engineering -> PCA whitening -> BIC selection -> walk-forward -> state relabeling -> hysteresis backtest -> performance metrics
- Accepts a `Config` object with all sub-configs
- Returns dict with: model, scaler, states, probabilities, equity_curve, positions, trades, metrics, bic_results, walk_forward_results
- Main entry point for the "best HMM" workflow

### Task 10: main.py — Integration
**File**: `main.py`  
**Changes**:
- Fix feature windows from window=3 to industry standards
- Add `--bic` flag for BIC-based state selection
- Add `--walk-forward` flag for walk-forward validation
- Add `--hysteresis` flag with configurable thresholds
- Add `--online-filter` flag for forward filter inference
- Keep backward compatibility with existing flags
- Update `add_features()` to use configurable windows

## Dependency Order
Phase 1 (parallel): Tasks 1, 2, 3, 4
Phase 2 (parallel, depends on config): Tasks 5, 6, 7, 8
Phase 3 (serial, depends on everything): Tasks 9, 10

## Success Criteria
- All existing tests pass: `python test_main.py && python test_lookahead.py`
- BIC selector returns different n_states for different datasets
- Walk-forward produces only out-of-sample predictions (no data leakage)
- Hysteresis reduces trade frequency vs no-hysteresis baseline
- OnlineHMMForwardFilter gives same results as batch predict_proba on in-sample data
- All new modules have docstrings and type hints
