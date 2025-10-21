Simple HMM Analysis
==================

A complete, step-by-step example of performing basic HMM analysis on futures data.

Overview
--------

This example walks through a complete HMM analysis pipeline:

1. Data loading and validation
2. Feature engineering
3. HMM model training
4. State inference
5. Basic visualization
6. Result interpretation

Prerequisites
------------

- Basic Python programming knowledge
- Understanding of pandas DataFrames
- Familiarity with financial concepts (OHLCV data)

Setup
-----

First, let's import the necessary libraries and create some sample data:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from pathlib import Path

   # Add src to path
   import sys
   sys.path.insert(0, str(Path('.').absolute() / 'src'))

   from data_processing.csv_parser import process_csv
   from data_processing.feature_engineering import add_features
   from model_training.hmm_trainer import train_model, validate_features_for_hmm
   from model_training.inference_engine import StateInference

   # Set style for plots
   plt.style.use('seaborn-v0_8')
   sns.set_palette("husl")

Step 1: Create Sample Data
---------------------------

For this example, we'll create synthetic futures data that resembles real market data:

.. code-block:: python

   def create_synthetic_futures_data(n_days=252):
       """Create synthetic futures data for demonstration."""
       np.random.seed(42)

       # Generate date range
       dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

       # Generate base price with trend and noise
       trend = np.linspace(100, 120, n_days)
       noise = np.random.normal(0, 2, n_days)
       prices = trend + noise

       # Generate OHLCV data
       high_spread = np.random.uniform(0.01, 0.03, n_days)
       low_spread = np.random.uniform(0.01, 0.03, n_days)

       data = pd.DataFrame({
           'datetime': dates,
           'open': prices,
           'high': prices * (1 + high_spread),
           'low': prices * (1 - low_spread),
           'close': prices + np.random.normal(0, 0.5, n_days),
           'volume': np.random.exponential(1, scale=10000, size=n_days)
       })

       return data

   # Create sample data
   data = create_synthetic_futures_data(252)  # 1 year of daily data
   print(f"Created synthetic data with {len(data)} rows")
   print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
   print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")

   # Save to CSV for later use
   data.to_csv('sample_futures_data.csv', index=False)
   print("Saved sample data to 'sample_futures_data.csv'")

Step 2: Load and Validate Data
---------------------------------

Load the data using our CSV parser:

.. code-block:: python

   # Load and process data
   processed_data = process_csv('sample_futures_data.csv')
   print(f"Loaded {len(processed_data)} rows of data")
   print(f"Columns: {list(processed_data.columns)}")

   # Display first few rows
   print("\nFirst 5 rows:")
   print(processed_data.head())

   # Basic data validation
   print(f"\nData info:")
   processed_data.info()

   # Check for missing values
   print(f"\nMissing values:\n{processed_data.isnull().sum()}")

Step 3: Feature Engineering
---------------------------

Add technical indicators that help the HMM identify market regimes:

.. code-block:: python

   # Add features using our feature engineering module
   features = add_features(processed_data)

   print(f"Added features. New shape: {features.shape}")
   print(f"Available feature columns: {[col for col in features.columns if col not in processed_data.columns]}")

   # Display engineered features
   print("\nSample engineered features:")
   print(features[['returns', 'volatility_14', 'sma_10', 'sma_20']].dropna().head(10))

   # Remove NaN values created by rolling windows
   features_clean = features.dropna()
   print(f"\nAfter removing NaN values: {len(features_clean)} rows remain")

Step 4: HMM Model Training
-------------------------

Train a Hidden Markov Model to identify market regimes:

.. code-block:: python

   # Prepare features for HMM (use close prices for simplicity)
   X = features_clean['close'].values.reshape(-1, 1)

   # Validate features
   validate_features_for_hmm(X)

   # Configure HMM model
   hmm_config = {
       'n_components': 3,  # Try 3 different market regimes
       'covariance_type': 'full',  # Full covariance matrix
       'n_iter': 100,  # Maximum iterations
       'random_state': 42,  # For reproducibility
   }

   # Train the model
   model, metadata = train_model(X, config=hmm_config)

   print(f"Trained HMM model:")
   print(f"  Number of states: {metadata['n_components']}")
   print(f"  Convergence: {metadata['converged']}")
   print(f"  Log-likelihood: {metadata['log_likelihood']:.2f}")
   print(f"  Training samples: {metadata['n_samples']}")

Step 5: State Inference
---------------------

Use the trained model to infer hidden states for the entire time series:

.. code-block:: python

   # Create inference engine
   inference = StateInference(model)

   # Infer states for all data
   states = inference.infer_states(X)

   print(f"Inferred states for {len(states)} time periods")
   print(f"Number of unique states: {len(np.unique(states))}")
   print(f"State distribution: {np.bincount(states) / len(states):.2%}")

   # Create a DataFrame with states
   results = features_clean.copy()
   results['hmm_state'] = states

   # Display state transitions
   print("\nFirst 20 state assignments:")
   print(results[['datetime', 'close', 'hmm_state']].head(20))

Step 6: Basic Visualization
-------------------------

Create visualizations to understand the results:

.. code-block:: python

   # Create figure with subplots
   fig, axes = plt.subplots(3, 1, figsize=(15, 12))
   fig.suptitle('HMM Analysis Results', fontsize=16, fontweight='bold')

   # Plot 1: Price with state colors
   ax1 = axes[0]
   for state in range(metadata['n_components']):
       state_mask = results['hmm_state'] == state
       ax1.scatter(
           results.loc[state_mask, 'datetime'],
           results.loc[state_mask, 'close'],
           label=f'State {state}',
           alpha=0.7,
           s=30
       )

   ax1.set_title('Price Data with HMM States')
   ax1.set_xlabel('Date')
   ax1.set_ylabel('Price ($)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Plot 2: State distribution over time
   ax2 = axes[1]
   ax2.plot(results['datetime'], results['hmm_state'],
            linewidth=1, alpha=0.8)
   ax2.set_title('State Sequence Over Time')
   ax2.set_xlabel('Date')
   ax2.set_ylabel('HMM State')
   ax2.set_yticks(range(metadata['n_components']))
   ax2.grid(True, alpha=0.3)

   # Plot 3: Returns by state
   ax3 = axes[2]
   for state in range(metadata['n_components']):
       state_returns = results.loc[results['hmm_state'] == state, 'returns']
       if len(state_returns) > 0:
           ax3.hist(state_returns, bins=20, alpha=0.6,
                    label=f'State {state}', density=True)

   ax3.set_title('Return Distribution by State')
   ax3.set_xlabel('Returns')
   ax3.set_ylabel('Density')
   ax3.legend()
   ax3.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Step 7: Analyze Results
--------------------

Calculate and interpret the results:

.. code-block:: python

   # State statistics
   print("\n=== State Analysis ===")

   for state in range(metadata['n_components']):
       state_data = results[results['hmm_state'] == state]
       if len(state_data) > 0:
           print(f"\nState {state}:")
           print(f"  Periods: {len(state_data)}")
           print(f"  Percentage: {len(state_data) / len(results):.1%}")
           print(f"  Mean return: {state_data['returns'].mean():.4f}")
           print(f"  Return std: {state_data['returns'].std():.4f}")
           print(f"  Volatility: {state_data['volatility_14'].mean():.4f}")

           # Calculate average duration
           state_changes = (state_data['hmm_state'] != state_data['hmm_state'].shift()).sum()
           avg_duration = len(state_data) / max(state_changes, 1)
           print(f"  Avg duration: {avg_duration:.1f} periods")

   # State transitions
   print("\n=== State Transition Analysis ===")
   transition_matrix = np.zeros((metadata['n_components'], metadata['n_components']))

   for i in range(len(states) - 1):
       from_state = states[i]
       to_state = states[i + 1]
       transition_matrix[from_state, to_state] += 1

   # Normalize transition matrix
   row_sums = transition_matrix.sum(axis=1)
   transition_matrix = transition_matrix / row_sums[:, np.newaxis]

   print("Transition Matrix (from\\to):")
   for i in range(metadata['n_components']):
       for j in range(metadata['n_components']):
           print(f"  {i} → {j}: {transition_matrix[i, j]:.3f}")

   # Model performance metrics
   print(f"\n=== Model Performance ===")
   print(f"Log-likelihood: {metadata['log_likelihood'].2f}")
   print(f"BIC: {metadata.get('bic', 'N/A')}")
   print(f"AIC: {metadata.get('aic', 'N/A')}")

Step 8: Save Results
-----------------

Save your analysis results for future reference:

.. code-block:: python

   # Save results to CSV
   results.to_csv('hmm_analysis_results.csv', index=False)
   print("Results saved to 'hmm_analysis_results.csv'")

   # Save model metadata
   import json
   with open('hmm_model_metadata.json', 'w') as f:
       json.dump(metadata, f, indent=2)
   print("Model metadata saved to 'hmm_model_metadata.json'")

   # Save configuration
   with open('hmm_config.json', 'w') as f:
       json.dump(hmm_config, f, indent=2)
   print("Configuration saved to 'hmm_config.json'")

   print("\n=== Analysis Complete ===")
   print("Files created:")
   print("  - hmm_analysis_results.csv (complete results)")
   print("  - hmm_model_metadata.json (model information)")
   print("  - hmm_config.json (training configuration)")
   print("  - sample_futures_data.csv (sample data)")

Next Steps
----------

Now that you have a basic understanding of HMM analysis, consider:

1. **Experiment with different numbers of states**
2. **Add more features to improve regime detection**
3. **Try different HMM configurations**
4. **Implement a simple trading strategy based on states**
5. **Test on real market data**

Advanced Topics
-------------

* **Multi-dimensional features**: Use multiple indicators instead of just close prices
* **Model selection**: Compare HMMs with different numbers of states
* **Cross-validation**: Use time series cross-validation for robustness
* **Ensemble methods**: Combine multiple HMMs for better predictions

Common Issues
------------

* **Insufficient data**: HMMs need sufficient data to learn meaningful patterns
* **Too many states**: Start with 2-3 states and increase gradually
* **Poor convergence**: Try different initializations or more iterations
* **Overfitting**: Use proper validation techniques

This example provides a solid foundation for understanding and using HMM analysis in futures markets.