Complete User Guide
==================

This comprehensive guide covers everything you need to know to effectively use the HMM Futures Analysis system for market regime detection and analysis.

Table of Contents
------------------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   quickstart
   concepts/hmm_basics
   concepts/market_regimes
   usage/data_preparation
   usage/configuration
   usage/cli_reference
   usage/advanced_usage
   examples/trading_strategies
   troubleshooting
   best_practices

Introduction
------------

The HMM Futures Analysis system is a comprehensive Python library designed to identify and analyze market regimes using Hidden Markov Models (HMMs). This powerful tool helps traders, analysts, and researchers:

* **Identify Market Regimes**: Automatically detect bull, bear, and sideways market conditions
* **Regime-Based Analysis**: Analyze market behavior within different regimes
* **Backtesting**: Test trading strategies across different market conditions
* **Risk Management**: Adjust risk parameters based on current market regime
* **Research**: Conduct quantitative research on market patterns

Key Features
~~~~~~~~~~~~

* **Multi-Engine Processing**: Support for streaming, Dask, and Daft processing engines
* **Advanced HMM Training**: Robust training with multiple restarts and convergence monitoring
* **Feature Engineering**: Comprehensive technical indicators and market features
* **Visualization**: Professional charts and interactive dashboards
* **CLI Interface**: Command-line tools for complete analysis workflows
* **Extensible Design**: Modular architecture for customization and extension

Who Should Use This Tool?
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Quantitative Analysts**: For market regime research and strategy development
* **Portfolio Managers**: For risk management and asset allocation decisions
* **Traders**: For market timing and strategy optimization
* **Researchers**: For academic studies and pattern analysis
* **Students**: For learning about machine learning in finance

System Requirements
-------------------

**Minimum Requirements:**

* Python 3.9 or higher
* 4GB RAM (8GB+ recommended for large datasets)
* 1GB disk space (2GB+ for full installation)

**Recommended Setup:**

* Python 3.11
* 8GB+ RAM
* SSD storage
* Linux or macOS (Windows supported but may require additional setup)

**Dependencies:**

Core dependencies are automatically installed, but key libraries include:

* `numpy`, `pandas`: Data manipulation and numerical computing
* `scikit-learn`, `hmmlearn`: Machine learning and HMM implementation
* `click`: Command-line interface
* `matplotlib`, `plotly`: Visualization
* `pytest`: Testing framework

Installation
------------

For detailed installation instructions, see the :doc:`installation` guide.

Quick Start
-----------

Once installed, you can start using the system immediately:

**Basic CLI Usage:**

.. code-block:: bash

   # Validate your data
   python cli_comprehensive.py validate -i your_data.csv

   # Run complete analysis
   python cli_comprehensive.py analyze -i your_data.csv -o results/

   # Get help
   python cli_comprehensive.py --help

**Python API Usage:**

.. code-block:: python

   from data_processing.csv_parser import process_csv
   from data_processing.feature_engineering import add_features
   from model_training.hmm_trainer import train_model

   # Load and process data
   data = process_csv('your_data.csv')
   features = add_features(data)

   # Train HMM model
   config = {'n_components': 3, 'n_iter': 100, 'random_state': 42}
   model, metadata = train_model(features, config)

Understanding HMMs
-----------------

What is a Hidden Markov Model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Hidden Markov Model is a statistical model that assumes:

1. **The system being modeled is a Markov process** - the future state depends only on the current state
2. **The states are hidden** - we can't directly observe which state the system is in
3. **Observations provide clues** - we can observe emissions that depend on the current state

In financial markets, this means:

* **Hidden States**: Market regimes (bull, bear, sideways)
* **Observations**: Price movements, volatility, volume, technical indicators
* **Transitions**: Probability of moving from one regime to another

Why HMMs for Financial Markets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HMMs are particularly well-suited for financial analysis because:

1. **Market Regimes Exist**: Markets exhibit distinct behaviors during different periods
2. **States are Not Directly Observable**: We can't definitively say "we are in a bull market" at any given moment
3. **Observable Data**: Prices, volume, and indicators provide clues about the current regime
4. **Regime Persistence**: Markets tend to stay in one regime for extended periods

Types of Regimes HMMs Can Detect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bull Markets:**
* Rising prices with moderate volatility
* High positive returns
* Increased trading volume
* Low fear metrics

**Bear Markets:**
* Falling prices with high volatility
* Negative returns
* High fear metrics (VIX, put/call ratios)
* Risk-off behavior

**Sideways Markets:**
* Price consolidation with low volatility
* Random walk behavior
* Low trading volume
* Neutral sentiment

**Transition Periods:**
* High volatility
* Rapid regime switches
* Uncertainty and mixed signals

Market Regimes in Detail
-----------------------

Characteristics of Each Regime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bull Market Regime:**

* **Price Behavior**: Sustained upward trend with minor pullbacks
* **Volatility**: Moderate and controlled
* **Volume**: High and increasing
* **Sentiment**: Optimistic, greedy
* **Indicators**: Rising moving averages, positive momentum
* **Duration**: Typically months to years in strong trends

**Bear Market Regime:**

* **Price Behavior**: Sustained downward trend with sharp rallies
* **Volatility**: High and erratic
* **Volume**: Spike during panic selling
* **Sentiment**: Pessimistic, fearful
* **Indicators**: Falling moving averages, negative momentum
* **Duration**: Typically shorter but more intense than bull markets

**Sideways/Neutral Regime:**

* **Price Behavior**: Range-bound, mean-reverting
* **Volatility**: Low and stable
* **Volume**: Low and consistent
* **Sentiment**: Neutral, uncertain
* **Indicators**: Flat moving averages, oscillating momentum
* **Duration**: Can last for extended periods

Regime Identification
~~~~~~~~~~~~~~~~~~

The HMM identifies regimes by analyzing:

1. **Return Patterns**: Statistical properties of price changes
2. **Volatility Clustering**: Periods of high/low volatility
3. **Momentum Persistence**: Trends in price movements
4. **Volume Patterns**: Trading volume characteristics
5. **Cross-Asset Relationships**: Correlations between different indicators

Data Preparation
-----------------

Supported Data Formats
~~~~~~~~~~~~~~~~~~~~

The system supports various CSV formats:

**Required Columns:**
- OHLC data: `open`, `high`, `low`, `close`
- Volume: `volume` (optional but recommended)

**Datetime Handling:**
- Automatic datetime column detection
- Support for various datetime formats
- Unix timestamp conversion
- Index-based datetime

**Column Name Variations:**
- Case insensitive: `Close`, `CLOSE`, `close`
- Common variations: `Close` vs `Price`, `Volume` vs `Vol`
- Underscore vs space: `open_price` vs `Open Price`

Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum Requirements:**

* **100+ data points**: Minimum for meaningful HMM training
* **No missing OHLC data**: Complete price history
* **Consistent time intervals**: Regular sampling (daily, hourly, etc.)

**Recommended Quality:**

* **1000+ data points**: Better for robust model training
* **Clean price data**: Remove outliers and errors
* **Consistent trading hours**: Avoid gaps in trading sessions
* **Sufficient volume data**: For better regime identification

**Data Validation:**

The system automatically validates data for:

* **Missing values**: Identifies and reports gaps
* **Data types**: Ensures numeric price data
* **Logical consistency**: OHLC relationships
* **Outliers**: Identifies extreme price movements
* **Date ranges**: Validates temporal continuity

Feature Engineering
-------------------

Built-in Features
~~~~~~~~~~~~~~~~

The system automatically calculates numerous technical indicators:

**Price-Based Features:**
- Log returns: `log_ret`, `simple_ret`
- Moving averages: `sma_5`, `sma_10`, `sma_20`, `ema_12`, `ema_26`
- Price position: `price_position_5`, `price_position_10`, `price_position_20`
- High-low ratios: `hl_ratio_5`, `hl_ratio_10`, `hl_ratio_20`

**Volatility Features:**
- Rolling volatility: `volatility_14`, `volatility_30`
- ATR (Average True Range): `atr_14`
- Bollinger Bands: `bb_upper`, `bb_lower`, `bb_width`, `bb_position`

**Momentum Features:**
- RSI (Relative Strength Index): `rsi_14`
- Rate of Change: `roc_10`, `roc_20`
- MACD: `macd`, `macd_signal`, `macd_histogram`

**Volume Features:**
- On-Balance Volume: `obv`
- Volume moving averages: `volume_sma_10`, `volume_sma_20`
- Volume price relationship: `vwap`

Custom Features
~~~~~~~~~~~~~~

You can add custom features by:

1. **Modifying the configuration file**:
   .. code-block:: yaml

      indicators:
        custom_indicator:
          window: 14
          parameter: value

2. **Extending the feature engineering module**:
   .. code-block:: python

      def custom_indicator(data, **kwargs):
          # Your custom calculation
          return result

Feature Selection
~~~~~~~~~~~~~~

**Best Features for HMM:**

1. **Log Returns**: Core statistical properties
2. **Volatility**: Regime-dependent volatility
3. **Momentum**: Trend strength and direction
4. **Price Position**: Relative to recent history
5. **Volume Patterns**: Market participation

**Avoid:**
- Highly correlated features (redundant information)
- Noisy indicators (too much randomness)
- Lookahead-biased features (future information)

Configuration
-------------

HMM Parameters
~~~~~~~~~~~~~~

**Core Parameters:**

* ``n_components``: Number of hidden states (2-5 recommended)
* ``covariance_type``: Shape of covariance matrix
  * ``full``: Most flexible, requires more data
  * ``diag``: Assumes independent features
  * ``tied``: Same covariance across states
  * ``spherical``: Same variance in all dimensions
* ``n_iter``: Maximum training iterations (100-500)
* ``tol``: Convergence tolerance (1e-3 to 1e-6)

**Advanced Parameters:**

* ``random_state``: Reproducibility (42 is common)
* ``num_restarts``: Multiple restarts for better convergence (3-10)
* ``init_params``: Initialization method
* ``algorithm``: Viterbi or MAP estimation

**Recommended Settings:**

.. code-block:: json

   {
     "n_components": 3,
     "covariance_type": "full",
     "n_iter": 100,
     "tol": 1e-3,
     "random_state": 42,
     "num_restarts": 3
   }

Processing Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Engine Types:**

* ``streaming``: Pandas-based, memory efficient (default)
* ``dask``: Parallel processing, large datasets
* ``daft``: Out-of-core, very large datasets

**Engine Selection Guide:**

* **< 100K rows**: Use ``streaming``
* **100K - 1M rows**: Use ``dask`` for parallel processing
* **> 1M rows**: Use ``daft`` for memory efficiency

**Configuration Example:**

.. code-block:: json

   {
     "engine_type": "dask",
     "chunk_size": 50000,
     "memory_limit": "8GB",
     "n_workers": 4
   }

CLI Reference
-------------

Command Structure
~~~~~~~~~~~~~~~~~

All commands follow the pattern:

.. code-block:: bash

   python cli_comprehensive.py [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

**Global Options:**

* ``--help``: Show help message
* ``--version``: Show version information
* ``--config-file PATH``: Use configuration file
* ``--log-level LEVEL``: Set logging level
* ``--memory-monitor``: Enable memory monitoring

**Available Commands:**

* ``validate``: Validate input data
* ``analyze``: Run complete HMM analysis
* ``infer``: Infer states on new data
* ``model-info``: Display model information
* ``version``: Show system information

Validate Command
~~~~~~~~~~~~~~~~

Validate data format and structure:

.. code-block:: bash

   # Basic validation
   python cli_comprehensive.py validate -i data.csv

   # With custom options
   python cli_comprehensive.py validate \
       -i data.csv \
       -o validation_output/ \
       --engine dask \
       --chunk-size 10000

**Options:**

* ``-i, --input-csv PATH``: Input CSV file (required)
* ``-o, --output-dir PATH``: Output directory
* ``--engine TYPE``: Processing engine
* ``--chunk-size INTEGER``: Chunk size for processing

Analyze Command
~~~~~~~~~~~~~~~~

Run complete HMM analysis pipeline:

.. code-block:: bash

   # Basic analysis
   python cli_comprehensive.py analyze -i data.csv -o results/

   # Advanced analysis
   python cli_comprehensive.py analyze \
       -i data.csv \
       -o results/ \
       --n-states 4 \
       --engine dask \
       --random-seed 123 \
       --test-size 0.2

**Options:**

* ``-i, --input-csv PATH``: Input CSV file (required)
* ``-o, --output-dir PATH``: Output directory (required)
* ``--n-states INTEGER``: Number of HMM states (default: 3)
* ``--engine TYPE``: Processing engine
* ``--test-size FLOAT``: Train/test split ratio
* ``--random-seed INTEGER``: Random seed for reproducibility

Infer Command
~~~~~~~~~~~~~~~

Load trained model and infer states:

.. code-block:: bash

   # Basic inference
   python cli_comprehensive.py infer \
       -m model.pkl \
       -i new_data.csv \
       -o inference_results/

   # With lag for bias prevention
   python cli_comprehensive.py infer \
       -m model.pkl \
       -i new_data.csv \
       -o results/ \
       --lag-periods 1

**Options:**

* ``-m, --model PATH``: Trained model file (required)
* ``-i, --input-csv PATH``: Input data (required)
* ``-o, --output-dir PATH``: Output directory
* ``--lag-periods INTEGER``: Lookahead bias prevention

Model-Info Command
~~~~~~~~~~~~~~~~~~

Display information about a trained model:

.. code-block:: bash

   python cli_comprehensive.py model-info -m model.pkl

**Options:**

* ``-m, --model PATH``: Model file (required)

Advanced Usage
-------------

Custom Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~

Create custom analysis workflows:

.. code-block:: python

   from utils import ProcessingConfig, HMMConfig
   from data_processing.csv_parser import process_csv
   from data_processing.feature_engineering import add_features
   from model_training.hmm_trainer import train_model
   from model_training.inference_engine import predict_states_comprehensive

   # Custom configuration
   processing_config = ProcessingConfig(
       engine_type='dask',
       chunk_size=100000,
       indicators={
           'custom_sma': {'window': 50},
           'custom_volatility': {'window': 30}
       }
   )

   hmm_config = HMMConfig(
       n_states=4,
       covariance_type='tied',
       n_iter=200,
       random_state=42
   )

   # Execute pipeline
   data = process_csv('data.csv', processing_config)
   features = add_features(data)

   # Select specific features
   selected_features = ['log_ret', 'volatility_14', 'rsi_14']
   feature_matrix = features[selected_features].dropna()

   # Train model
   model, metadata = train_model(feature_matrix, hmm_config.to_dict())

   # Inference
   results = predict_states_comprehensive(
       model, metadata['scaler'], feature_matrix.values, selected_features
   )

Regime-Based Trading Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy Framework:**

1. **Identify Current Regime**: Use HMM to determine market state
2. **Select Strategy**: Choose appropriate strategy for current regime
3. **Risk Management**: Adjust position sizing based on regime volatility
4. **Performance Monitoring**: Track strategy performance by regime

**Example Implementation:**

.. code-block:: python

   def regime_based_strategy(hmm_states, prices, signals):
       """Implement different strategies based on HMM states"""

       positions = pd.Series(index=prices.index, data=0)

       for i, state in enumerate(hmm_states):
           if state == 0:  # Bull market
               # Trend-following strategy
               if signals['momentum'][i] > 0:
                   positions.iloc[i] = 1
               else:
                   positions.iloc[i] = 0

           elif state == 1:  # Bear market
               # Defensive strategy
               positions.iloc[i] = -0.5  # Short position

           else:  # Sideways market
               # Mean reversion strategy
               if signals['rsi'][i] < 30:
                   positions.iloc[i] = 1
               elif signals['rsi'][i] > 70:
                   positions.iloc[i] = -1
               else:
                   positions.iloc[i] = 0

       return positions

Model Validation
~~~~~~~~~~~~~~~

**Cross-Validation Approach:**

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit

   # Time series cross-validation
   tscv = TimeSeriesSplit(n_splits=5)

   scores = []
   for train_idx, test_idx in tscv.split(feature_matrix):
       train_data = feature_matrix.iloc[train_idx]
       test_data = feature_matrix.iloc[test_idx]

       # Train on train_data
       model, metadata = train_model(train_data, hmm_config)

       # Evaluate on test_data
       test_results = predict_states_comprehensive(
           model, metadata['scaler'], test_data.values, selected_features
       )

       # Calculate score (e.g., log likelihood per sample)
       score = test_results.log_likelihood / len(test_data)
       scores.append(score)

**Performance Metrics:**

* **Log Likelihood**: Model fit quality
* **AIC/BIC**: Model comparison metrics
* **Regime Persistence**: Stability of identified states
* **Transition Probability**: Reasonableness of state transitions
* **Economic Significance**: Interpretability of regimes

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Memory Errors:**

* **Symptom**: OutOfMemoryError during processing
* **Solution**:
  - Use Dask engine for large datasets
  - Reduce chunk size
  - Increase system memory
  - Use streaming engine for memory efficiency

**Convergence Issues:**

* **Symptom**: Model fails to converge or poor log likelihood
* **Solution**:
  - Increase ``n_iter`` parameter
  - Try different ``covariance_type``
  - Scale features properly
  - Use multiple restarts (``num_restarts``)
  - Check data quality

**Poor Regime Identification:**

* **Symptom**: All observations classified as single state
* **Solution**:
  - Check feature relevance
  - Try different number of states
  - Ensure sufficient data diversity
  - Verify data preprocessing

**Import Errors:**

* **Symptom**: ModuleNotFoundError or ImportError
* **Solution**:
  - Ensure proper Python path setup
  - Install missing dependencies
  - Check virtual environment activation

Getting Help
~~~~~~~~~~

**Debug Mode:**

.. code-block:: bash

   # Enable debug logging
   python cli_comprehensive.py --log-level DEBUG analyze -i data.csv

**Community Support:**

* **GitHub Issues**: Report bugs and feature requests
* **Documentation**: Check this guide and API reference
* **Examples**: Review example notebooks and scripts

Best Practices
--------------

Data Management
~~~~~~~~~~~~~~

**Data Quality:**
- Always validate input data before analysis
- Handle missing values appropriately
- Remove or correct obvious data errors
- Ensure consistent time intervals

**Feature Engineering:**
- Start with basic features, add complexity gradually
- Avoid features with lookahead bias
- Normalize or scale features appropriately
- Remove highly correlated features

**Model Training:**
- Use multiple random seeds for robustness
- Validate with out-of-sample data
- Monitor convergence carefully
- Compare models with different parameters

**Performance:**
- Use appropriate processing engines for data size
- Monitor memory usage during large analyses
- Cache intermediate results when possible
- Profile code for optimization opportunities

Analysis Workflow
~~~~~~~~~~~~~~~~~

**Recommended Pipeline:**

1. **Data Exploration**: Understand your data characteristics
2. **Preprocessing**: Clean and prepare data
3. **Feature Selection**: Choose relevant indicators
4. **Baseline Model**: Start with simple configuration
5. **Iteration**: Refine features and parameters
6. **Validation**: Test on out-of-sample data
7. **Documentation**: Record methodology and results

**Reproducibility:**
- Set random seeds consistently
- Save model configurations
- Document data sources and preprocessing
- Version control analysis scripts
- Store intermediate results

Risk Management
~~~~~~~~~~~~~~

**Model Risk:**
- Never trust a single model completely
- Use ensemble approaches when possible
- Validate across different market conditions
- Monitor model performance over time

**Market Risk:**
- HMM states are probabilistic, not deterministic
- Past regime patterns may not repeat
- Black swan events can break all models
- Always use fundamental analysis alongside models

**Implementation Risk:**
- Test thoroughly before live deployment
- Start with paper trading
- Monitor for model degradation
- Have fallback strategies ready

This complete guide should help you get the most out of the HMM Futures Analysis system. Happy analyzing! 🚀