# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
This is a Hidden Markov Model (HMM) futures trading analysis project with three distinct implementations for handling large CSV files.

## Non-Obvious Project Patterns

### Data Processing Approaches
- **Three different HMM implementations**: [`main.py`](main.py:1) uses pandas streaming, [`hmm_futures_script.py`](hmm_futures_script.py:1) uses Dask, and [`hmm_futures_daft.py`](hmm_futures_daft.py:1) uses Daft - each for different memory/scaling requirements
- **Feature engineering differs by implementation**: main.py uses log-returns + ATR + ROC, script.py uses log-returns + range, daft.py uses log-returns + volatility proxy
- **Memory optimization**: Script version downcasts to float32 (lines 75-77), Daft version uses Arrow backend for lazy evaluation

### Critical Implementation Details
- **State interpretation**: All implementations use the same regime logic - State 0 = long, State 1 = flat, State 2 = short (not obvious from code structure)
- **Position shifting**: Script version has lookahead bias prevention via [`np.roll(positions, 1)`](hmm_futures_script.py:85), but main.py version doesn't implement this
- **Index alignment**: Script version manually aligns data after feature engineering with [`df_big.iloc[len(df_big) - len(states):]`](hmm_futures_script.py:121)

### Performance Considerations
- **Chunking strategy**: main.py uses 100k row chunks, but doesn't handle memory-constrained environments
- **Daft dtype helper**: Custom [`dtype_fix_dict()`](hmm_futures_daft.py:39) function required for Daft compatibility
- **Model persistence**: Script and Daft versions save both model AND scaler together, main.py saves neither

### Backtesting Gotchas
- **Sharpe calculation**: Uses 252*78 multiplier assuming intraday data (lines 65, 95) - this assumption is buried in the code
- **Performance metrics**: Different implementations calculate drawdown differently (min vs cumulative approaches)
- **Plotting**: matplotlib is optional in main.py but required in other versions

## Commands
```bash
# Install dependencies (uses uv package manager)
uv sync

# Run basic HMM (pandas streaming)
python main.py data.csv --n_states 3 --plot

# Run Dask version (memory efficient)
python hmm_futures_script.py data.csv --symbol ES --model-out model.pkl

# Run Daft version (lazy evaluation)
python hmm_futures_daft.py data.csv --states 3 --model-out bundle.pkl
```

## Dependencies
- Python 3.13+ required (specified in [.python-version](.python-version:1))
- Uses `uv` for dependency management (not pip/poetry)
- Heavy ML stack: scikit-learn, hmmlearn, pandas, numpy, matplotlib