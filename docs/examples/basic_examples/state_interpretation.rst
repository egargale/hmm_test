State Interpretation
====================

Understanding and interpreting the hidden states identified by HMM models.

Overview
--------

Hidden Markov Models identify underlying market regimes, but interpreting what these states represent is crucial for practical application. This guide covers systematic approaches to state interpretation, validation, and practical usage.

State Characteristics
--------------------

Statistical Properties
~~~~~~~~~~~~~~~~~~~~~

Each hidden state has distinct statistical characteristics that help identify market regimes:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns

   def analyze_state_characteristics(data_with_states):
       """Comprehensive analysis of HMM state characteristics."""

       # Basic statistics by state
       state_stats = {}
       for state in data_with_states['hmm_state'].unique():
           state_data = data_with_states[data_with_states['hmm_state'] == state]

           state_stats[state] = {
               'count': len(state_data),
               'percentage': len(state_data) / len(data_with_states) * 100,
               'mean_return': state_data['returns'].mean(),
               'std_return': state_data['returns'].std(),
               'volatility': state_data['volatility_14'].mean(),
               'mean_volume': state_data['volume'].mean(),
               'price_trend': state_data['close'].pct_change(20).mean(),
               'max_drawdown': calculate_max_drawdown(state_data['close']),
               'sharpe_ratio': state_data['returns'].mean() / state_data['returns'].std() * np.sqrt(252)
           }

       # Convert to DataFrame for easier analysis
       stats_df = pd.DataFrame(state_stats).T
       print("State Characteristics:")
       print(stats_df.round(4))

       return stats_df

   def calculate_max_drawdown(prices):
       """Calculate maximum drawdown for a price series."""
       peak = prices.expanding().max()
       drawdown = (prices - peak) / peak
       return drawdown.min()

   # Analyze states
   state_stats = analyze_state_characteristics(results)

Visual State Profiles
~~~~~~~~~~~~~~~~~~~~~

Create visual profiles for each state:

.. code-block:: python

   def create_state_profiles(data_with_states, stats_df):
       """Create comprehensive visual profiles for each state."""

       n_states = len(stats_df)
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       fig.suptitle('HMM State Profiles', fontsize=16, fontweight='bold')

       # Plot 1: Returns distribution
       ax1 = axes[0, 0]
       for state in stats_df.index:
           state_returns = data_with_states[data_with_states['hmm_state'] == state]['returns']
           ax1.hist(state_returns, bins=30, alpha=0.6,
                   label=f'State {state}', density=True)
       ax1.set_title('Return Distribution by State')
       ax1.set_xlabel('Returns')
       ax1.set_ylabel('Density')
       ax1.legend()
       ax1.grid(True, alpha=0.3)

       # Plot 2: Volatility comparison
       ax2 = axes[0, 1]
       vol_by_state = [data_with_states[data_with_states['hmm_state'] == state]['volatility_14'].mean()
                      for state in stats_df.index]
       bars = ax2.bar(stats_df.index, vol_by_state, alpha=0.7)
       ax2.set_title('Average Volatility by State')
       ax2.set_xlabel('State')
       ax2.set_ylabel('Volatility')
       ax2.grid(True, alpha=0.3)

       # Color bars by volatility level
       for bar, vol in zip(bars, vol_by_state):
           if vol < vol_by_state[np.median(np.arange(len(vol_by_state)))]:
               bar.set_color('green')
           elif vol > vol_by_state[np.percentile(np.arange(len(vol_by_state)), 75)]:
               bar.set_color('red')
           else:
               bar.set_color('orange')

       # Plot 3: State duration analysis
       ax3 = axes[0, 2]
       durations = calculate_state_durations(data_with_states['hmm_state'])
       duration_boxes = [durations[state] for state in stats_df.index if state in durations]
       state_labels = [f'State {state}' for state in stats_df.index if state in durations]

       ax3.boxplot(duration_boxes, labels=state_labels)
       ax3.set_title('State Duration Distribution')
       ax3.set_xlabel('State')
       ax3.set_ylabel('Duration (periods)')
       ax3.grid(True, alpha=0.3)

       # Plot 4: Risk-return scatter
       ax4 = axes[1, 0]
       scatter = ax4.scatter(stats_df['std_return'], stats_df['mean_return'],
                           c=stats_df.index, s=100, alpha=0.7, cmap='viridis')
       ax4.set_xlabel('Risk (Std Return)')
       ax4.set_ylabel('Mean Return')
       ax4.set_title('Risk-Return Profile by State')
       ax4.grid(True, alpha=0.3)

       # Add state labels
       for i, state in enumerate(stats_df.index):
           ax4.annotate(f'State {state}',
                       (stats_df.loc[state, 'std_return'], stats_df.loc[state, 'mean_return']),
                       xytext=(5, 5), textcoords='offset points')

       # Plot 5: Volume characteristics
       ax5 = axes[1, 1]
       volume_by_state = [data_with_states[data_with_states['hmm_state'] == state]['volume'].mean()
                         for state in stats_df.index]
       ax5.bar(stats_df.index, volume_by_state, alpha=0.7, color='purple')
       ax5.set_title('Average Volume by State')
       ax5.set_xlabel('State')
       ax5.set_ylabel('Volume')
       ax5.grid(True, alpha=0.3)

       # Plot 6: Transition frequency heatmap
       ax6 = axes[1, 2]
       transition_matrix = calculate_transition_matrix(data_with_states['hmm_state'])
       im = ax6.imshow(transition_matrix, cmap='Blues', aspect='auto')
       ax6.set_title('State Transition Matrix')
       ax6.set_xlabel('To State')
       ax6.set_ylabel('From State')

       # Add text annotations
       for i in range(len(transition_matrix)):
           for j in range(len(transition_matrix)):
               ax6.text(j, i, f'{transition_matrix[i, j]:.3f}',
                       ha='center', va='center', color='white' if transition_matrix[i, j] > 0.5 else 'black')

       plt.colorbar(im, ax=ax6)
       plt.tight_layout()
       plt.show()

   def calculate_state_durations(state_series):
       """Calculate duration of consecutive state occurrences."""
       durations = {}

       for state in state_series.unique():
           state_mask = (state_series == state)
           # Find consecutive groups
           groups = (state_mask != state_mask.shift()).cumsum()
           durations_list = state_mask.groupby(groups).sum()
           # Only keep actual durations (when state_mask is True)
           durations[state] = durations_list[durations_list > 0].values

       return durations

   # Create visual profiles
   create_state_profiles(results, state_stats)

Regime Classification
---------------------

Automatic Labeling
~~~~~~~~~~~~~~~~~~~

Develop an automated system to label states based on their characteristics:

.. code-block:: python

   def classify_market_regimes(state_stats, thresholds=None):
       """Classify HMM states into market regimes."""
       if thresholds is None:
           thresholds = {
               'low_vol_threshold': 0.02,
               'high_vol_threshold': 0.05,
               'trend_threshold': 0.001,
               'return_threshold': 0.0005
           }

       regimes = {}

       for state, stats in state_stats.iterrows():
           volatility = stats['volatility']
           trend = stats['price_trend']
           mean_return = stats['mean_return']

           # Classification logic
           if volatility < thresholds['low_vol_threshold']:
               if abs(trend) < thresholds['trend_threshold']:
                   regimes[state] = 'Ranging_Low_Vol'
               elif trend > thresholds['trend_threshold']:
                   regimes[state] = 'Trending_Up_Low_Vol'
               else:
                   regimes[state] = 'Trending_Down_Low_Vol'

           elif volatility > thresholds['high_vol_threshold']:
               if mean_return > thresholds['return_threshold']:
                   regimes[state] = 'Bullish_High_Vol'
               elif mean_return < -thresholds['return_threshold']:
                   regimes[state] = 'Bearish_High_Vol'
               else:
                   regimes[state] = 'Volatile_Transitional'

           else:
               if mean_return > thresholds['return_threshold']:
                   regimes[state] = 'Moderate_Bull'
               elif mean_return < -thresholds['return_threshold']:
                   regimes[state] = 'Moderate_Bear'
               else:
                   regimes[state] = 'Transitional'

       return regimes

   def detailed_regime_analysis(data_with_states, regimes):
       """Detailed analysis of each regime type."""
       regime_analysis = {}

       for state, regime_type in regimes.items():
           state_data = data_with_states[data_with_states['hmm_state'] == state]

           regime_analysis[regime_type] = {
               'state_id': state,
               'periods': len(state_data),
               'percentage': len(state_data) / len(data_with_states) * 100,
               'mean_return': state_data['returns'].mean(),
               'volatility': state_data['volatility_14'].mean(),
               'max_drawdown': calculate_max_drawdown(state_data['close']),
               'volume_ratio': state_data['volume'].mean() / data_with_states['volume'].mean(),
               'up_capture': (state_data['returns'] > 0).mean(),
               'down_capture': (state_data['returns'] < 0).mean()
           }

       return regime_analysis

   # Classify regimes
   regimes = classify_market_regimes(state_stats)
   regime_analysis = detailed_regime_analysis(results, regimes)

   print("\nRegime Classification:")
   for regime, analysis in regime_analysis.items():
       print(f"\n{regime}:")
       for key, value in analysis.items():
           print(f"  {key}: {value:.4f}")

Dynamic Regime Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate regime classifications over time:

.. code-block:: python

   def validate_regime_stability(data_with_states, window=60):
       """Validate the stability of regime classifications over time."""
       stability_metrics = {}

       for state in data_with_states['hmm_state'].unique():
           state_data = data_with_states[data_with_states['hmm_state'] == state]

           # Rolling statistics for this state
           rolling_returns = state_data['returns'].rolling(window).mean()
           rolling_vol = state_data['volatility_14'].rolling(window).mean()

           stability_metrics[state] = {
               'return_stability': rolling_returns.std(),
               'volatility_stability': rolling_vol.std(),
               'consistency_score': calculate_consistency_score(state_data, window)
           }

       return stability_metrics

   def calculate_consistency_score(state_data, window):
       """Calculate how consistent the state characteristics are over time."""
       if len(state_data) < window:
           return 0.0

       # Calculate rolling z-scores for key metrics
       rolling_mean_return = state_data['returns'].rolling(window).mean()
       rolling_vol = state_data['volatility_14'].rolling(window).mean()

       # Consistency is inverse of variation
       mean_consistency = 1 / (1 + rolling_mean_return.std())
       vol_consistency = 1 / (1 + rolling_vol.std())

       return (mean_consistency + vol_consistency) / 2

   # Validate regime stability
   stability_metrics = validate_regime_stability(results)

State Transitions
-----------------

Transition Analysis
~~~~~~~~~~~~~~~~~~~

Analyze how states transition and what drives transitions:

.. code-block:: python

   def analyze_state_transitions(state_series, data):
       """Comprehensive analysis of state transitions."""

       # Calculate transition matrix
       n_states = len(state_series.unique())
       transition_matrix = np.zeros((n_states, n_states))

       for i in range(len(state_series) - 1):
           from_state = state_series.iloc[i]
           to_state = state_series.iloc[i + 1]
           transition_matrix[from_state, to_state] += 1

       # Normalize to probabilities
       row_sums = transition_matrix.sum(axis=1, keepdims=True)
       transition_probs = transition_matrix / np.where(row_sums > 0, row_sums, 1)

       # Analyze transition drivers
       transition_analysis = {}

       for from_state in range(n_states):
           for to_state in range(n_states):
               if transition_matrix[from_state, to_state] > 0:
                   # Find periods when this transition occurred
                   transition_periods = []
                   for i in range(len(state_series) - 1):
                       if state_series.iloc[i] == from_state and state_series.iloc[i + 1] == to_state:
                           transition_periods.append(i)

                   if transition_periods:
                       # Analyze market conditions at transitions
                       pre_transition_data = data.iloc[transition_periods]

                       transition_analysis[f"{from_state}->{to_state}"] = {
                           'count': len(transition_periods),
                           'frequency': transition_matrix[from_state, to_state],
                           'probability': transition_probs[from_state, to_state],
                           'avg_return_before': pre_transition_data['returns'].mean(),
                           'avg_vol_before': pre_transition_data['volatility_14'].mean(),
                           'volume_spike': pre_transition_data['volume'].mean() / data['volume'].mean()
                       }

       return transition_probs, transition_analysis

   def visualize_transition_patterns(transition_probs, transition_analysis):
       """Visualize state transition patterns."""

       fig, axes = plt.subplots(1, 3, figsize=(18, 6))
       fig.suptitle('State Transition Analysis', fontsize=16, fontweight='bold')

       # Plot 1: Transition matrix heatmap
       ax1 = axes[0]
       im1 = ax1.imshow(transition_probs, cmap='YlOrRd', aspect='auto')
       ax1.set_title('Transition Probability Matrix')
       ax1.set_xlabel('To State')
       ax1.set_ylabel('From State')

       # Add probability values
       for i in range(len(transition_probs)):
           for j in range(len(transition_probs)):
               if transition_probs[i, j] > 0.01:  # Only show meaningful probabilities
                   ax1.text(j, i, f'{transition_probs[i, j]:.3f}',
                           ha='center', va='center',
                           color='white' if transition_probs[i, j] > 0.5 else 'black')

       plt.colorbar(im1, ax=ax1)

       # Plot 2: Transition frequency
       ax2 = axes[1]
       transitions = list(transition_analysis.keys())
       frequencies = [analysis['count'] for analysis in transition_analysis.values()]

       bars = ax2.bar(range(len(transitions)), frequencies)
       ax2.set_title('Transition Frequency')
       ax2.set_xlabel('Transition Type')
       ax2.set_ylabel('Frequency')
       ax2.set_xticks(range(len(transitions)))
       ax2.set_xticklabels(transitions, rotation=45, ha='right')
       ax2.grid(True, alpha=0.3)

       # Plot 3: Volume impact on transitions
       ax3 = axes[2]
       volume_impacts = [analysis['volume_spike'] for analysis in transition_analysis.values()]

       bars = ax3.bar(range(len(transitions)), volume_impacts, alpha=0.7, color='purple')
       ax3.set_title('Volume Impact on Transitions')
       ax3.set_xlabel('Transition Type')
       ax3.set_ylabel('Volume Ratio')
       ax3.set_xticks(range(len(transitions)))
       ax3.set_xticklabels(transitions, rotation=45, ha='right')
       ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Normal Volume')
       ax3.grid(True, alpha=0.3)
       ax3.legend()

       plt.tight_layout()
       plt.show()

   # Analyze transitions
   transition_probs, transition_analysis = analyze_state_transitions(results['hmm_state'], results)
   visualize_transition_patterns(transition_probs, transition_analysis)

Predicting Transitions
~~~~~~~~~~~~~~~~~~~~~

Build models to predict state transitions:

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report, confusion_matrix

   def predict_state_transitions(data_with_states, lookback_period=10):
       """Predict state transitions using machine learning."""

       # Create transition labels
       transitions = []
       for i in range(lookback_period, len(data_with_states) - 1):
           current_state = data_with_states['hmm_state'].iloc[i]
           next_state = data_with_states['hmm_state'].iloc[i + 1]
           transitions.append(f"{current_state}_to_{next_state}")

       # Create features for prediction
       features = []
       for i in range(lookback_period, len(data_with_states) - 1):
           window_data = data_with_states.iloc[i-lookback_period:i]

           feature_vector = [
               window_data['returns'].mean(),
               window_data['returns'].std(),
               window_data['volatility_14'].iloc[-1],
               window_data['volume'].mean(),
               window_data['close'].pct_change(5).iloc[-1],
               window_data['high'].max() - window_data['low'].min(),  # Range
               window_data['close'].iloc[-1] / window_data['close'].iloc[0] - 1,  # Cumulative return
               len(window_data[window_data['volume'] > window_data['volume'].mean()]) / lookback_period  # High volume days
           ]

           features.append(feature_vector)

       # Prepare data for ML
       X = np.array(features)
       y = np.array(transitions)

       # Only keep transitions that occur frequently enough
       transition_counts = pd.Series(y).value_counts()
       common_transitions = transition_counts[transition_counts >= 5].index

       mask = pd.Series(y).isin(common_transitions)
       X_filtered = X[mask]
       y_filtered = y[mask]

       # Split data
       X_train, X_test, y_train, y_test = train_test_split(
           X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
       )

       # Train model
       rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
       rf.fit(X_train, y_train)

       # Evaluate
       y_pred = rf.predict(X_test)
       print("Transition Prediction Results:")
       print(classification_report(y_test, y_pred))

       # Feature importance
       feature_names = [
           'mean_return', 'return_std', 'current_vol', 'mean_volume',
           'short_momentum', 'price_range', 'cumulative_return', 'high_volume_ratio'
       ]

       feature_importance = pd.DataFrame({
           'feature': feature_names,
           'importance': rf.feature_importances_
       }).sort_values('importance', ascending=False)

       print("\nFeature Importance for Transition Prediction:")
       print(feature_importance)

       return rf, feature_importance

   # Predict transitions
   transition_model, feature_importance = predict_state_transitions(results)

Practical Applications
---------------------

Trading Strategy Design
~~~~~~~~~~~~~~~~~~~~~~~

Use state interpretations to design trading strategies:

.. code-block:: python

   def design_regime_based_strategy(data_with_states, regimes):
       """Design trading strategies based on regime interpretations."""

       strategy_signals = pd.DataFrame(index=data_with_states.index)
       strategy_signals['regime'] = data_with_states['hmm_state'].map(regimes)
       strategy_signals['returns'] = data_with_states['returns']
       strategy_signals['close'] = data_with_states['close']

       # Strategy rules based on regime
       for regime_type in regimes.values():
           if 'Bullish' in regime_type:
               # Trend-following strategies
               strategy_signals.loc[strategy_signals['regime'] == regime_type, 'signal'] = \
                   (strategy_signals['close'] > strategy_signals['close'].rolling(10).mean()).astype(int)

           elif 'Bearish' in regime_type:
               # Defensive/cash strategies
               strategy_signals.loc[strategy_signals['regime'] == regime_type, 'signal'] = 0

           elif 'Ranging' in regime_type:
               # Mean reversion strategies
               bb_upper = strategy_signals['close'].rolling(20).mean() + 2 * strategy_signals['close'].rolling(20).std()
               bb_lower = strategy_signals['close'].rolling(20).mean() - 2 * strategy_signals['close'].rolling(20).std()

               signals = pd.Series(0, index=strategy_signals.index)
               signals[strategy_signals['close'] < bb_lower] = 1  # Buy at lower band
               signals[strategy_signals['close'] > bb_upper] = -1  # Sell at upper band

               strategy_signals.loc[strategy_signals['regime'] == regime_type, 'signal'] = signals

           else:
               # Default conservative approach
               strategy_signals.loc[strategy_signals['regime'] == regime_type, 'signal'] = 0

       # Calculate strategy returns
       strategy_signals['strategy_returns'] = strategy_signals['signal'].shift(1) * strategy_signals['returns']

       # Performance analysis by regime
       regime_performance = {}
       for regime_type in strategy_signals['regime'].unique():
           if pd.isna(regime_type):
               continue

           regime_data = strategy_signals[strategy_signals['regime'] == regime_type]
           if len(regime_data) > 0:
               regime_performance[regime_type] = {
                   'total_return': regime_data['strategy_returns'].sum(),
                   'sharpe_ratio': regime_data['strategy_returns'].mean() / regime_data['strategy_returns'].std() * np.sqrt(252) if regime_data['strategy_returns'].std() > 0 else 0,
                   'max_drawdown': calculate_max_drawdown(regime_data['strategy_returns'].cumsum()),
                   'win_rate': (regime_data['strategy_returns'] > 0).mean(),
                   'periods': len(regime_data)
               }

       return strategy_signals, regime_performance

   # Design and test strategy
   strategy_signals, regime_performance = design_regime_based_strategy(results, regimes)

   print("\nStrategy Performance by Regime:")
   for regime, performance in regime_performance.items():
       print(f"\n{regime}:")
       for metric, value in performance.items():
           print(f"  {metric}: {value:.4f}")

Risk Management
~~~~~~~~~~~~~~~

Use state interpretations for risk management:

.. code-block:: python

   def regime_based_risk_management(data_with_states, regimes):
       """Implement risk management based on regime characteristics."""

       risk_parameters = pd.DataFrame(index=data_with_states.index)
       risk_parameters['regime'] = data_with_states['hmm_state'].map(regimes)

       # Calculate regime-specific risk metrics
       regime_risk = {}
       for regime_type in regimes.values():
           regime_data = data_with_states[data_with_states['hmm_state'].map(regimes) == regime_type]

           if len(regime_data) > 10:
               regime_risk[regime_type] = {
                   'volatility': regime_data['returns'].std(),
                   'var_95': regime_data['returns'].quantile(0.05),
                   'max_drawdown': calculate_max_drawdown(regime_data['close']),
                   'downside_frequency': (regime_data['returns'] < 0).mean()
               }

       # Set risk limits based on regime
       for regime_type, risk_metrics in regime_risk.items():
           base_position_size = 1.0

           # Adjust position size based on volatility
           if risk_metrics['volatility'] < 0.02:  # Low volatility
               volatility_adjustment = 1.2
           elif risk_metrics['volatility'] > 0.05:  # High volatility
               volatility_adjustment = 0.5
           else:
               volatility_adjustment = 1.0

           # Adjust for downside risk
           if risk_metrics['downside_frequency'] > 0.6:  # High downside frequency
               downside_adjustment = 0.7
           elif risk_metrics['downside_frequency'] < 0.4:  # Low downside frequency
               downside_adjustment = 1.1
           else:
               downside_adjustment = 1.0

           # Calculate final position size
           position_size = base_position_size * volatility_adjustment * downside_adjustment
           position_size = min(position_size, 1.5)  # Cap at 150%
           position_size = max(position_size, 0.1)  # Minimum 10%

           risk_parameters.loc[risk_parameters['regime'] == regime_type, 'position_size'] = position_size

           # Set stop-loss levels
           if risk_metrics['volatility'] > 0.04:
               stop_loss = 0.03  # 3% stop loss for high volatility
           elif risk_metrics['volatility'] < 0.015:
               stop_loss = 0.01  # 1% stop loss for low volatility
           else:
               stop_loss = 0.02  # 2% default

           risk_parameters.loc[risk_parameters['regime'] == regime_type, 'stop_loss'] = stop_loss

       return risk_parameters, regime_risk

   # Implement risk management
   risk_params, regime_risk = regime_based_risk_management(results, regimes)

   print("\nRegime Risk Characteristics:")
   for regime, risk in regime_risk.items():
       print(f"\n{regime}:")
       for metric, value in risk.items():
           print(f"  {metric}: {value:.4f}")

This comprehensive guide to state interpretation provides the foundation for understanding HMM results and translating them into practical trading and risk management decisions.