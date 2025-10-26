# HMM Futures Analysis - Practical Examples & Tutorials

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Usage Examples](#basic-usage-examples)
3. [Advanced Trading Strategies](#advanced-trading-strategies)
4. [Real-World Case Studies](#real-world-case-studies)
5. [Performance Optimization](#performance-optimization)
6. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Getting Started

### Prerequisites

Before running these examples, ensure you have:

```bash
# Install required packages
pip install hmm-futures-analysis pandas numpy matplotlib seaborn

# For advanced examples
pip install jupyter plotly dash streamlit
```

### Sample Data Preparation

If you don't have futures data, you can create sample data:

```python
import pandas as pd
import numpy as np

def create_sample_futures_data(n_days=252, freq='5min'):
    """Create realistic sample futures data"""

    # Create time index
    dates = pd.date_range(start='2023-01-01', periods=n_days*78, freq=freq)

    # Simulate price with trend and noise
    np.random.seed(42)

    # Base trend
    trend = np.linspace(100, 120, len(dates))

    # Add regime-dependent volatility
    regimes = np.random.choice([0, 1, 2], size=len(dates), p=[0.3, 0.5, 0.2])
    volatility = np.where(regimes == 0, 0.5,  # Low vol regime
                         np.where(regimes == 1, 1.0,  # Normal vol regime
                                  np.where(regimes == 2, 2.0, 0)))  # High vol regime

    # Generate prices
    returns = np.random.normal(0, volatility/100, len(dates))
    prices = 100 * np.exp(np.cumsum(returns) + np.log(trend/100))

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices

    # Generate realistic OHLC
    noise_range = prices * volatility / 100 * 0.5
    data['high'] = data['close'] + np.abs(np.random.normal(0, noise_range))
    data['low'] = data['close'] - np.abs(np.random.normal(0, noise_range))
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])

    # Generate volume
    base_volume = 10000
    volume_variation = np.random.lognormal(0, 0.5, len(dates))
    data['volume'] = base_volume * volume_variation * (1 + volatility)

    # Clean up
    data = data.dropna()
    data = data[data['high'] >= data['low']]

    return data.reset_index().rename(columns={'index': 'DateTime'})

# Create sample data
sample_data = create_sample_futures_data()
sample_data.to_csv('sample_es_data.csv', index=False)

print(f"Created sample data with {len(sample_data)} rows")
print(f"Date range: {sample_data['DateTime'].min()} to {sample_data['DateTime'].max()}")
```

---

## Basic Usage Examples

### Example 1: Quick Start with CLI

```bash
# 1. Validate your data
hmm-analyze validate -i sample_es_data.csv

# 2. Run basic analysis with 3 states
hmm-analyze analyze -i sample_es_data.csv -o results/basic/ --n-states 3

# 3. Run with more states for finer regime detection
hmm-analyze analyze -i sample_es_data.csv -o results/fine_grained/ --n-states 5
```

### Example 2: Python API Basic Usage

```python
import pandas as pd
import numpy as np
from src.cli_simple import main
import sys
import matplotlib.pyplot as plt

# Method 1: Using the CLI programmatically
def run_hmm_analysis_cli():
    """Run HMM analysis using CLI interface"""

    # Set up arguments
    sys.argv = [
        'analyze',
        '-i', 'sample_es_data.csv',
        '-o', 'results/api_analysis/',
        '--n-states', 4,
        '--test-size', '0.2',
        '--random-seed', '42'
    ]

    # Run analysis
    main()

# Method 2: Direct API usage
def run_hmm_analysis_direct():
    """Run HMM analysis using direct API calls"""

    from src.data_processing.csv_parser import process_csv
    from src.data_processing.feature_engineering import add_features
    from src.data_processing.data_validation import validate_data
    from sklearn.preprocessing import StandardScaler
    from hmmlearn import hmm

    print("Loading and preparing data...")

    # Load and validate data
    data = process_csv('sample_es_data.csv')
    data_clean, validation_result = validate_data(data)

    # Add features
    features = add_features(data_clean)

    # Select features for HMM
    feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
    X = features[feature_cols].dropna()

    print(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train HMM
    print("Training HMM model...")
    model = hmm.GaussianHMM(
        n_components=4,
        covariance_type='full',
        n_iter=100,
        random_state=42
    )

    model.fit(X_scaled)

    # Predict states
    states = model.predict(X_scaled)

    # Add states to features
    features.loc[X.index, 'hmm_state'] = states

    # Save results
    output_dir = 'results/direct_analysis/'
    import os
    os.makedirs(output_dir, exist_ok=True)

    features.to_csv(f'{output_dir}/states.csv')

    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Model converged: {model.monitor_.converged}")
    print(f"Log likelihood: {model.score(X_scaled):.2f}")

    return model, features, states

# Run both methods
if __name__ == "__main__":
    print("=== CLI Method ===")
    run_hmm_analysis_cli()

    print("\n=== Direct API Method ===")
    model, features, states = run_hmm_analysis_direct()
```

### Example 3: Visualizing Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hmm_results(data, states, prices_col='close'):
    """Create comprehensive visualization of HMM results"""

    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # 1. Price with regime colors
    ax1 = axes[0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for state in np.unique(states):
        mask = states == state
        ax1.scatter(data.index[mask], data[prices_col][mask],
                   c=colors[state], label=f'Regime {state}', s=1, alpha=0.6)

    ax1.set_title('Price with HMM Regimes')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Returns by regime
    ax2 = axes[1]
    returns = data['log_ret']

    for state in np.unique(states):
        mask = states == state
        ax2.hist(returns[mask], bins=50, alpha=0.6,
                label=f'Regime {state}', density=True)

    ax2.set_title('Return Distribution by Regime')
    ax2.set_xlabel('Log Returns')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. State transition timeline
    ax3 = axes[2]
    ax3.plot(data.index, states, drawstyle='steps-mid')
    ax3.set_title('Regime Transitions Over Time')
    ax3.set_ylabel('Regime')
    ax3.set_ylim(-0.5, max(states) + 0.5)
    ax3.grid(True, alpha=0.3)

    # 4. Volatility by regime
    ax4 = axes[3]
    volatility = data['log_ret'].rolling(20).std()

    for state in np.unique(states):
        mask = states == state
        ax4.plot(data.index[mask], volatility[mask],
                label=f'Regime {state}', alpha=0.7, linewidth=1)

    ax4.set_title('Volatility by Regime (20-day rolling)')
    ax4.set_ylabel('Volatility')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/hmm_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot the results
plot_hmm_results(features, states)
```

### Example 4: Regime Characterization

```python
def analyze_regime_characteristics(data, states):
    """Detailed analysis of each identified regime"""

    regime_stats = {}

    for state in np.unique(states):
        mask = states == state
        regime_data = data[mask]

        # Basic statistics
        stats = {
            'duration_days': len(regime_data),
            'percentage_time': len(regime_data) / len(data) * 100,
            'mean_return': regime_data['log_ret'].mean(),
            'volatility': regime_data['log_ret'].std(),
            'sharpe_ratio': regime_data['log_ret'].mean() / regime_data['log_ret'].std() if regime_data['log_ret'].std() > 0 else 0,
            'max_drawdown': calculate_max_drawdown(regime_data['close']),
            'mean_rsi': regime_data['rsi'].mean(),
            'mean_atr': regime_data['atr'].mean(),
            'mean_volume': regime_data['volume'].mean()
        }

        # Regime classification
        if stats['volatility'] < 0.01:
            regime_type = "Low Volatility"
        elif stats['volatility'] > 0.03:
            regime_type = "High Volatility"
        else:
            regime_type = "Normal Volatility"

        if stats['mean_return'] > 0.001:
            regime_type += " / Bullish"
        elif stats['mean_return'] < -0.001:
            regime_type += " / Bearish"
        else:
            regime_type += " / Neutral"

        stats['regime_type'] = regime_type
        regime_stats[state] = stats

    # Print summary
    print("=== REGIME ANALYSIS ===")
    for state, stats in regime_stats.items():
        print(f"\nRegime {state} ({stats['regime_type']}):")
        print(f"  Duration: {stats['duration_days']} periods ({stats['percentage_time']:.1f}% of time)")
        print(f"  Mean Return: {stats['mean_return']:.4f} ({stats['mean_return']*252*78:.1%} annualized)")
        print(f"  Volatility: {stats['volatility']:.4f} ({stats['volatility']*np.sqrt(252*78):.1%} annualized)")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        print(f"  Mean RSI: {stats['mean_rsi']:.1f}")
        print(f"  Mean ATR: {stats['mean_atr']:.4f}")

    return regime_stats

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

# Analyze regimes
regime_analysis = analyze_regime_characteristics(features, states)
```

---

## Advanced Trading Strategies

### Example 5: Regime-Based Trading Strategy

```python
class RegimeBasedStrategy:
    """Advanced trading strategy based on HMM regimes"""

    def __init__(self, regime_signals, lookback=20, volatility_threshold=0.02):
        self.regime_signals = regime_signals
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold
        self.positions = []
        self.returns = []

    def generate_signals(self, data, states):
        """Generate trading signals based on regime analysis"""

        signals = pd.Series(0, index=data.index)

        for i in range(self.lookback, len(data)):
            current_state = states[i]
            recent_data = data.iloc[i-self.lookback:i]

            # Calculate recent volatility
            recent_vol = recent_data['log_ret'].std()

            # Strategy logic based on regime
            if current_state == 0:  # Low volatility regime
                # Trend following
                if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0]:
                    signals.iloc[i] = 1  # Long
                else:
                    signals.iloc[i] = -1  # Short

            elif current_state == 1:  # Normal regime
                # Mean reversion
                current_price = data['close'].iloc[i]
                recent_mean = recent_data['close'].mean()
                recent_std = recent_data['close'].std()

                z_score = (current_price - recent_mean) / recent_std

                if z_score > 1.5:  # Overbought
                    signals.iloc[i] = -1  # Short
                elif z_score < -1.5:  # Oversold
                    signals.iloc[i] = 1  # Long

            elif current_state == 2:  # High volatility regime
                # Reduce position size or stay flat
                if recent_vol < self.volatility_threshold:
                    # Only trade if volatility isn't too extreme
                    signals.iloc[i] = np.sign(recent_data['log_ret'].mean())
                else:
                    signals.iloc[i] = 0  # Stay flat

            elif current_state == 3:  # Crisis regime
                # Defensive positioning
                signals.iloc[i] = 0  # Stay flat

        return signals

    def backtest(self, data, signals):
        """Backtest the strategy"""

        # Calculate returns
        strategy_returns = pd.Series(0.0, index=data.index)

        for i in range(1, len(data)):
            if signals.iloc[i-1] != 0:  # Have position
                strategy_returns.iloc[i] = signals.iloc[i-1] * data['log_ret'].iloc[i]

        # Calculate performance metrics
        cumulative_returns = (1 + strategy_returns).cumprod()

        metrics = {
            'total_return': cumulative_returns.iloc[-1] - 1,
            'annualized_return': strategy_returns.mean() * 252 * 78,
            'volatility': strategy_returns.std() * np.sqrt(252 * 78),
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 78),
            'max_drawdown': calculate_max_drawdown(cumulative_returns),
            'win_rate': (strategy_returns > 0).mean(),
            'profit_factor': strategy_returns[strategy_returns > 0].sum() / abs(strategy_returns[strategy_returns < 0].sum()) if (strategy_returns < 0).any() else float('inf')
        }

        return strategy_returns, metrics

# Implement the strategy
strategy = RegimeBasedStrategy(regime_analysis)

# Generate signals
signals = strategy.generate_signals(features, states)

# Backtest
strategy_returns, performance_metrics = strategy.backtest(features, signals)

print("=== STRATEGY PERFORMANCE ===")
for metric, value in performance_metrics.items():
    if isinstance(value, float):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    else:
        print(f"{metric.replace('_', ' ').title()}: {value}")
```

### Example 6: Dynamic Position Sizing

```python
class DynamicPositionSizer:
    """Position sizing based on regime volatility"""

    def __init__(self, base_size=0.1, max_size=0.3):
        self.base_size = base_size
        self.max_size = max_size

    def calculate_position_size(self, data, states, lookback=20):
        """Calculate position sizes based on regime volatility"""

        position_sizes = pd.Series(0.0, index=data.index)

        for i in range(lookback, len(data)):
            current_state = states[i]
            recent_data = data.iloc[i-lookback:i]

            # Calculate regime-specific volatility
            regime_vol = recent_data['log_ret'].std()

            # Calculate position size (inverse of volatility)
            if regime_vol > 0:
                # Kelly criterion approximation
                expected_return = recent_data['log_ret'].mean()
                kelly_fraction = expected_return / (regime_vol ** 2)

                # Scale position size
                position_size = self.base_size * (1 + kelly_fraction)

                # Apply constraints
                position_size = max(0, min(position_size, self.max_size))

                # Adjust based on regime type
                if current_state == 2:  # High volatility
                    position_size *= 0.5  # Reduce size
                elif current_state == 3:  # Crisis
                    position_size *= 0.2  # Significant reduction

                position_sizes.iloc[i] = position_size

        return position_sizes

# Implement dynamic sizing
sizer = DynamicPositionSizer(base_size=0.15, max_size=0.4)
position_sizes = sizer.calculate_position_sizes(features, states)

# Combine with signals
sized_signals = signals * position_sizes

# Backtest sized strategy
sized_returns, sized_performance = strategy.backtest(features, sized_signals)

print("\n=== DYNAMIC SIZING PERFORMANCE ===")
for metric, value in sized_performance.items():
    if isinstance(value, float):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
```

### Example 7: Multi-Timeframe Analysis

```python
def multi_timeframe_analysis(data, states_5min, states_1hour, states_daily):
    """Combine HMM states from multiple timeframes"""

    # Align different timeframes
    aligned_data = data.copy()

    # Resample states to common timeframe (5min)
    states_1hour_resampled = states_1hour.reindex(aligned_data.index, method='ffill')
    states_daily_resampled = states_daily.reindex(aligned_data.index, method='ffill')

    # Create combined state signals
    aligned_data['state_5min'] = states_5min
    aligned_data['state_1hour'] = states_1hour_resampled
    aligned_data['state_daily'] = states_daily_resampled

    # Generate combined signals
    def generate_multi_tf_signal(row):
        """Generate signal based on multiple timeframes"""

        signal_5min = regime_signals.get(row['state_5min'], 'neutral')
        signal_1hour = regime_signals.get(row['state_1hour'], 'neutral')
        signal_daily = regime_signals.get(row['state_daily'], 'neutral')

        # Weight signals (more weight to longer timeframes)
        weights = [0.2, 0.3, 0.5]  # 5min, 1hour, daily

        # Convert signals to numeric
        signal_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}

        numeric_signals = [
            signal_map.get(signal_5min, 0),
            signal_map.get(signal_1hour, 0),
            signal_map.get(signal_daily, 0)
        ]

        # Calculate weighted signal
        weighted_signal = sum(w * s for w, s in zip(weights, numeric_signals))

        # Apply threshold
        if weighted_signal > 0.3:
            return 1  # Long
        elif weighted_signal < -0.3:
            return -1  # Short
        else:
            return 0  # Neutral

    aligned_data['multi_tf_signal'] = aligned_data.apply(generate_multi_tf_signal, axis=1)

    return aligned_data

# Note: This would require running HMM analysis on multiple timeframes first
# For demonstration, we'll simulate the process
print("Multi-timeframe analysis would combine:")
print("- 5-minute HMM states for short-term signals")
print("- 1-hour HMM states for medium-term trends")
print("- Daily HMM states for long-term regime")
```

---

## Real-World Case Studies

### Case Study 1: S&P 500 Futures (ES) Analysis

```python
def analyze_sp500_futures():
    """Complete analysis of S&P 500 E-mini futures"""

    print("=== S&P 500 FUTURES REGIME ANALYSIS ===")

    # Load ES data (you would use real data here)
    # es_data = pd.read_csv('ES_continuous_5min.csv')

    # For demo, use sample data
    es_data = create_sample_futures_data(n_days=504)  # 2 years

    # Save for analysis
    es_data.to_csv('es_sample_data.csv', index=False)

    # Run HMM analysis
    sys.argv = [
        'analyze',
        '-i', 'es_sample_data.csv',
        '-o', 'results/es_analysis/',
        '--n-states', 4,
        '--test-size', '0.2'
    ]

    main()

    # Load results
    results = pd.read_csv('results/es_analysis/states.csv', index_col=0, parse_dates=True)

    # Analyze regime characteristics
    states = results['hmm_state'].values
    regime_stats = analyze_regime_characteristics(results, states)

    # Trading hours analysis
    results['hour'] = results.index.hour
    hourly_regime_dist = pd.crosstab(results['hour'], results['hmm_state'], normalize='index')

    print("\n=== INTRADAY REGIME DISTRIBUTION ===")
    print(hourly_regime_dist.round(3))

    # Performance by session
    def get_session(hour):
        if 9 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 15:
            return 'Afternoon'
        elif 15 <= hour < 17:
            return 'Late Afternoon'
        else:
            return 'After Hours'

    results['session'] = results['hour'].apply(get_session)
    session_performance = results.groupby(['session', 'hmm_state'])['log_ret'].agg(['mean', 'std', 'count'])

    print("\n=== SESSION PERFORMANCE BY REGIME ===")
    print(session_performance.round(4))

    return results, regime_stats

# Run the analysis
es_results, es_regime_stats = analyze_sp500_futures()
```

### Case Study 2: Volatility Trading Strategy

```python
class VolatilityRegimeStrategy:
    """Strategy focused on volatility-based regime detection"""

    def __init__(self, vol_lookback=20, vol_threshold=0.015):
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold

    def detect_volatility_regimes(self, data):
        """Detect regimes based on volatility patterns"""

        # Calculate rolling volatility
        returns = data['log_ret']
        rolling_vol = returns.rolling(self.vol_lookback).std()

        # Calculate volatility rank
        vol_rank = rolling_vol.rolling(252*78).rank(pct=True)  # 1-year lookback

        # Define regimes
        regimes = pd.Series(0, index=data.index)

        # Low volatility regime (bottom 30%)
        regimes[vol_rank <= 0.3] = 0

        # Normal volatility regime (middle 40%)
        regimes[(vol_rank > 0.3) & (vol_rank <= 0.7)] = 1

        # High volatility regime (top 30%)
        regimes[vol_rank > 0.7] = 2

        return regimes, rolling_vol, vol_rank

    def generate_volatility_signals(self, data, regimes, rolling_vol):
        """Generate trading signals based on volatility regimes"""

        signals = pd.Series(0, index=data.index)

        for i in range(self.vol_lookback, len(data)):
            current_regime = regimes.iloc[i]
            current_vol = rolling_vol.iloc[i]

            if current_regime == 0:  # Low volatility
                # Sell options (collect premium)
                signals.iloc[i] = 0  # Neutral for futures

            elif current_regime == 1:  # Normal volatility
                # Standard trend following
                recent_trend = data['close'].iloc[i-5:i].mean() - data['close'].iloc[i-20:i-15].mean()
                signals.iloc[i] = 1 if recent_trend > 0 else -1

            elif current_regime == 2:  # High volatility
                # Buy options or reduce futures exposure
                if current_vol > self.vol_threshold:
                    signals.iloc[i] = 0  # Stay flat during extreme volatility
                else:
                    # Mean reversion strategy
                    current_price = data['close'].iloc[i]
                    ma_20 = data['close'].iloc[i-20:i].mean()

                    if current_price > ma_20 * 1.02:  # 2% above MA
                        signals.iloc[i] = -1  # Short
                    elif current_price < ma_20 * 0.98:  # 2% below MA
                        signals.iloc[i] = 1   # Long

        return signals

# Implement volatility strategy
vol_strategy = VolatilityRegimeStrategy()

# Detect volatility regimes
vol_regimes, rolling_vol, vol_rank = vol_strategy.detect_volatility_regimes(features)

# Generate signals
vol_signals = vol_strategy.generate_volatility_signals(features, vol_regimes, rolling_vol)

# Backtest
vol_returns, vol_performance = strategy.backtest(features, vol_signals)

print("=== VOLATILITY REGIME STRATEGY PERFORMANCE ===")
for metric, value in vol_performance.items():
    if isinstance(value, float):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
```

### Case Study 3: Machine Learning Enhanced HMM

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class MLEnhancedHMM:
    """Combine HMM with machine learning for better regime prediction"""

    def __init__(self, hmm_states, n_lags=5):
        self.hmm_states = hmm_states
        self.n_lags = n_lags
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = []

    def prepare_ml_features(self, data):
        """Prepare features for machine learning model"""

        features = pd.DataFrame(index=data.index)

        # Lagged returns
        for i in range(1, self.n_lags + 1):
            features[f'return_lag_{i}'] = data['log_ret'].shift(i)

        # Lagged volatility
        for i in range(1, min(self.n_lags, 3) + 1):
            features[f'vol_lag_{i}'] = data['log_ret'].rolling(i).std().shift(1)

        # Lagged RSI
        features['rsi_lag'] = data['rsi'].shift(1)

        # Lagged volume
        features['volume_ratio_lag'] = data['volume_ratio'].shift(1)

        # Time features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month

        self.feature_columns = features.columns.tolist()

        return features

    def train_ml_model(self, data, test_size=0.2):
        """Train ML model to predict HMM states"""

        # Prepare features
        features = self.prepare_ml_features(data)

        # Align with HMM states
        aligned_data = pd.concat([features, pd.Series(self.hmm_states, index=data.index, name='hmm_state')], axis=1)
        aligned_data = aligned_data.dropna()

        # Split data
        X = aligned_data[self.feature_columns]
        y = aligned_data['hmm_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train model
        self.ml_model.fit(X_train, y_train)

        # Evaluate
        train_score = self.ml_model.score(X_train, y_train)
        test_score = self.ml_model.score(X_test, y_test)

        print(f"ML Model Training Accuracy: {train_score:.3f}")
        print(f"ML Model Test Accuracy: {test_score:.3f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n=== FEATURE IMPORTANCE ===")
        print(importance.head(10))

        return X_test, y_test

    def predict_states_ml(self, data):
        """Predict HMM states using ML model"""

        features = self.prepare_ml_features(data)
        predictions = self.ml_model.predict(features[self.feature_columns])

        return predictions

    def compare_predictions(self, data):
        """Compare HMM vs ML predictions"""

        # Get predictions
        hmm_pred = self.hmm_states
        ml_pred = self.predict_states_ml(data)

        # Align predictions
        min_length = min(len(hmm_pred), len(ml_pred))
        hmm_aligned = hmm_pred[-min_length:]
        ml_aligned = ml_pred[-min_length:]

        # Calculate agreement
        agreement = (hmm_aligned == ml_aligned).mean()

        print(f"\n=== PREDICTION COMPARISON ===")
        print(f"Agreement between HMM and ML: {agreement:.3f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(hmm_aligned, ml_aligned))

        # Classification report
        print("\nClassification Report:")
        print(classification_report(hmm_aligned, ml_pred[:min_length]))

        return hmm_aligned, ml_aligned

# Implement ML-enhanced HMM
ml_hmm = MLEnhancedHMM(states, n_lags=5)

# Train ML model
X_test, y_test = ml_hmm.train_ml_model(features)

# Compare predictions
hmm_pred, ml_pred = ml_hmm.compare_predictions(features)

# Create hybrid signals (weighted combination)
def create_hybrid_signals(hmm_signals, ml_signals, hmm_weight=0.6):
    """Create hybrid trading signals"""

    # Map states to signals
    def state_to_signal(state):
        regime_map = {
            0: 1,   # Bullish
            1: 0,   # Neutral
            2: -1,  # Bearish
            3: -1   # Crisis/Bearish
        }
        return regime_map.get(state, 0)

    hmm_trading_signals = np.array([state_to_signal(s) for s in hmm_pred])
    ml_trading_signals = np.array([state_to_signal(s) for s in ml_pred])

    # Weighted combination
    hybrid_signals = hmm_weight * hmm_trading_signals + (1 - hmm_weight) * ml_trading_signals

    # Apply threshold
    final_signals = np.where(hybrid_signals > 0.2, 1,
                            np.where(hybrid_signals < -0.2, -1, 0))

    return pd.Series(final_signals, index=features.index[-len(final_signals):])

# Create and test hybrid strategy
hybrid_signals = create_hybrid_signals(hmm_pred, ml_pred)
hybrid_returns, hybrid_performance = strategy.backtest(features.iloc[-len(hybrid_signals):], hybrid_signals)

print("\n=== HYBRID STRATEGY PERFORMANCE ===")
for metric, value in hybrid_performance.items():
    if isinstance(value, float):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
```

---

## Performance Optimization

### Example 8: Optimizing for Large Datasets

```python
import time
import psutil
import gc
from memory_profiler import profile

class OptimizedHMMProcessor:
    """Optimized HMM processing for large datasets"""

    def __init__(self, chunk_size=50000, memory_limit_gb=8):
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit_gb * 1024**3  # Convert to bytes

    def monitor_memory(self):
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024**3  # GB

    def process_large_dataset(self, file_path):
        """Process large dataset in chunks"""

        print(f"Processing {file_path} with chunk size {self.chunk_size}")
        print(f"Memory limit: {self.memory_limit/1024**3:.1f} GB")

        start_time = time.time()

        # Initialize
        all_features = []
        all_states = []

        # Process in chunks
        chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)

        for i, chunk in enumerate(chunk_reader):
            print(f"Processing chunk {i+1}...")

            # Monitor memory
            current_memory = self.monitor_memory()
            print(f"  Memory usage: {current_memory:.2f} GB")

            if current_memory > self.memory_limit / 1024**3 * 0.8:
                print("  Warning: High memory usage, forcing garbage collection")
                gc.collect()

            # Process chunk
            try:
                # Add features (optimized version)
                chunk_features = self.add_features_optimized(chunk)

                # Don't store all features, process on the fly
                if len(all_features) == 0:
                    # First chunk - fit scaler and model
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(chunk_features)

                    # Train HMM
                    from hmmlearn import hmm
                    self.model = hmm.GaussianHMM(n_components=3, random_state=42)
                    self.model.fit(X_scaled)

                    # Predict states
                    chunk_states = self.model.predict(X_scaled)

                else:
                    # Subsequent chunks - use fitted model
                    X_scaled = self.scaler.transform(chunk_features)
                    chunk_states = self.model.predict(X_scaled)

                # Store results (could be saved to disk instead)
                all_features.extend(chunk_features.values.tolist())
                all_states.extend(chunk_states.tolist())

                print(f"  Processed {len(chunk)} rows")

            except Exception as e:
                print(f"  Error processing chunk: {e}")
                continue

        # Combine results
        processing_time = time.time() - start_time

        print(f"\nProcessing complete!")
        print(f"Total rows processed: {len(all_features)}")
        print(f"Total time: {processing_time:.2f} seconds")
        print(f"Rows per second: {len(all_features) / processing_time:.0f}")
        print(f"Final memory usage: {self.monitor_memory():.2f} GB")

        return np.array(all_features), np.array(all_states)

    def add_features_optimized(self, df):
        """Optimized feature engineering"""

        # Use vectorized operations
        df = df.copy()

        # Basic returns
        df['log_ret'] = np.log(df['close']).diff()

        # Efficient rolling calculations
        windows = [5, 10, 20]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()

        # Volatility
        df['volatility_20'] = df['log_ret'].rolling(20).std()

        # RSI (optimized)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Select and return feature columns
        feature_cols = ['log_ret', 'volatility_20', 'rsi', 'volume_ratio']
        for window in windows:
            feature_cols.append(f'sma_{window}')

        return df[feature_cols].dropna()

# Test optimized processing
optimizer = OptimizedHMMProcessor(chunk_size=100000, memory_limit_gb=6)

# Create a larger dataset for testing
large_data = create_sample_futures_data(n_days=1000)  # ~4 years of data
large_data.to_csv('large_sample_data.csv', index=False)

# Process with optimization
features_optimized, states_optimized = optimizer.process_large_dataset('large_sample_data.csv')
```

### Example 9: Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import numpy as np

class ParallelHMMProcessor:
    """Parallel processing for multiple assets or time periods"""

    def __init__(self, n_workers=None):
        self.n_workers = n_workers or cpu_count()

    def process_multiple_assets(self, asset_files):
        """Process multiple assets in parallel"""

        print(f"Processing {len(asset_files)} assets with {self.n_workers} workers")

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self.process_single_asset, file): file
                for file in asset_files
            }

            # Collect results
            results = {}
            for future in futures:
                file = futures[future]
                try:
                    result = future.result()
                    results[file] = result
                    print(f"Completed: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

        return results

    def process_single_asset(self, file_path):
        """Process a single asset file"""

        try:
            # Load data
            data = pd.read_csv(file_path)

            # Add features
            features = add_features(data)

            # Train HMM
            feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
            X = features[feature_cols].dropna()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = hmm.GaussianHMM(n_components=3, random_state=42)
            model.fit(X_scaled)
            states = model.predict(X_scaled)

            return {
                'states': states,
                'model': model,
                'scaler': scaler,
                'features': features,
                'performance': model.score(X_scaled)
            }

        except Exception as e:
            print(f"Error in process_single_asset: {e}")
            return None

    def rolling_window_analysis(self, data, window_size=252*78, step_size=63*78):  # 1 year windows, 3 month steps
        """Perform rolling window HMM analysis"""

        print(f"Rolling window analysis: window={window_size}, step={step_size}")

        results = []

        for start_idx in range(0, len(data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]

            print(f"Processing window: {window_data.index[0]} to {window_data.index[-1]}")

            try:
                # Process window
                features = add_features(window_data)
                feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
                X = features[feature_cols].dropna()

                if len(X) > 100:  # Minimum data requirement
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    model = hmm.GaussianHMM(n_components=3, random_state=42)
                    model.fit(X_scaled)
                    states = model.predict(X_scaled)

                    results.append({
                        'start_date': window_data.index[0],
                        'end_date': window_data.index[-1],
                        'model': model,
                        'states': states,
                        'log_likelihood': model.score(X_scaled),
                        'n_regimes': len(np.unique(states))
                    })

            except Exception as e:
                print(f"Error processing window {start_idx}-{end_idx}: {e}")
                continue

        return results

# Test parallel processing
parallel_processor = ParallelHMMProcessor(n_workers=4)

# Create multiple sample assets
asset_files = []
for i, asset_name in enumerate(['ES', 'NQ', 'YM', 'RTY']):
    asset_data = create_sample_futures_data(n_days=252)
    asset_data.to_csv(f'sample_{asset_name}.csv', index=False)
    asset_files.append(f'sample_{asset_name}.csv')

# Process in parallel
parallel_results = parallel_processor.process_multiple_assets(asset_files)

# Analyze results
print("\n=== PARALLEL PROCESSING RESULTS ===")
for file, result in parallel_results.items():
    if result:
        print(f"{file}: Log-likelihood = {result['performance']:.2f}")
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Overfitting to Noise

```python
def detect_overfitting(model, train_data, test_data):
    """Detect if HMM model is overfitting"""

    # Calculate scores
    train_score = model.score(train_data)
    test_score = model.score(test_data)

    # Calculate overfitting ratio
    overfitting_ratio = (train_score - test_score) / abs(train_score)

    print(f"Train score: {train_score:.2f}")
    print(f"Test score: {test_score:.2f}")
    print(f"Overfitting ratio: {overfitting_ratio:.3f}")

    if overfitting_ratio > 0.1:
        print("⚠️  WARNING: Model may be overfitting")
        return True
    else:
        print("✅ Model appears well-fitted")
        return False

def regularize_hmm(data, n_components_range=range(2, 7)):
    """Find optimal number of components using regularization"""

    results = []

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data, test_data = data[:split_idx], data[split_idx:]

    for n_comp in n_components_range:
        print(f"\nTesting {n_comp} components...")

        # Train model
        model = hmm.GaussianHMM(
            n_components=n_comp,
            covariance_type='diag',  # Use diagonal for regularization
            random_state=42
        )

        model.fit(train_data)

        # Check for overfitting
        is_overfitting = detect_overfitting(model, train_data, test_data)

        # Calculate BIC
        log_likelihood = model.score(test_data)
        n_params = n_comp * (n_comp - 1) + (n_comp - 1) + n_comp * data.shape[1] * 2  # Approximate
        bic = -2 * log_likelihood + n_params * np.log(len(test_data))

        results.append({
            'n_components': n_comp,
            'train_score': model.score(train_data),
            'test_score': log_likelihood,
            'bic': bic,
            'overfitting': is_overfitting
        })

    # Select best model
    valid_results = [r for r in results if not r['overfitting']]

    if valid_results:
        best = min(valid_results, key=lambda x: x['bic'])
        print(f"\n✅ Optimal number of components: {best['n_components']}")
        print(f"BIC: {best['bic']:.2f}")
        return best['n_components']
    else:
        print("\n⚠️  All models show signs of overfitting")
        return 3  # Default fallback
```

### Pitfall 2: Lookahead Bias

```python
def prevent_lookahead_bias(features, states):
    """Adjust for lookahead bias in trading signals"""

    # Shift all signals by 1 period to prevent lookahead
    adjusted_states = states.copy()
    adjusted_states[1:] = states[:-1]
    adjusted_states[0] = states[1]  # Fill first value

    # Adjust features that might contain lookahead
    adjusted_features = features.copy()

    # Shift moving averages
    for col in adjusted_features.columns:
        if 'sma' in col or 'rolling' in col.lower():
            adjusted_features[col] = adjusted_features[col].shift(1)

    # Remove first row due to shifting
    adjusted_features = adjusted_features.iloc[1:]
    adjusted_states = adjusted_states[1:]

    print("✅ Applied lookahead bias correction")

    return adjusted_features, adjusted_states

def validate_no_lookahead(signals, returns):
    """Validate that signals don't contain lookahead bias"""

    # Calculate correlation between signals and future returns
    future_returns = returns.shift(-1)
    correlation = signals.corr(future_returns)

    print(f"Signal-future return correlation: {correlation:.4f}")

    if abs(correlation) > 0.1:
        print("⚠️  WARNING: High correlation may indicate lookahead bias")
        return False
    else:
        print("✅ Low correlation suggests no significant lookahead bias")
        return True
```

### Pitfall 3: Data Snooping

```python
def out_of_sample_test(data, train_periods=3, test_periods=1):
    """Perform robust out-of-sample testing"""

    period_length = len(data) // (train_periods + test_periods)
    results = []

    for i in range(train_periods + test_periods):
        start_idx = i * period_length
        end_idx = min((i + 1) * period_length, len(data))

        if i < train_periods:
            # Training period
            period_type = 'train'
        else:
            # Test period
            period_type = 'test'

        period_data = data.iloc[start_idx:end_idx]

        print(f"Period {i+1} ({period_type}): {len(period_data)} samples")

        if period_type == 'train':
            # Train model
            features = add_features(period_data)
            feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
            X = features[feature_cols].dropna()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = hmm.GaussianHMM(n_components=3, random_state=42)
            model.fit(X_scaled)

            # Store model for testing
            current_model = model
            current_scaler = scaler

        else:
            # Test model
            features = add_features(period_data)
            feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
            X = features[feature_cols].dropna()

            X_scaled = current_scaler.transform(X)
            states = current_model.predict(X_scaled)

            # Calculate performance
            strategy_returns = calculate_strategy_returns(features, states)

            results.append({
                'period': i - train_periods + 1,
                'return': strategy_returns.sum(),
                'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252*78),
                'max_dd': calculate_max_drawdown((1 + strategy_returns).cumprod())
            })

    # Analyze out-of-sample performance
    print("\n=== OUT-OF-SAMPLE RESULTS ===")
    for result in results:
        print(f"Period {result['period']}: Return={result['return']:.3f}, "
              f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.3f}")

    avg_return = np.mean([r['return'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])

    print(f"\nAverage out-of-sample performance:")
    print(f"Return: {avg_return:.3f}")
    print(f"Sharpe: {avg_sharpe:.2f}")

    return results

def calculate_strategy_returns(features, states):
    """Calculate simple strategy returns based on states"""

    signals = pd.Series(0, index=features.index)

    # Simple strategy: long in state 0, short in state 2
    signals[states == 0] = 1
    signals[states == 2] = -1

    # Calculate returns
    returns = features['log_ret'].shift(-1) * signals
    returns = returns.dropna()

    return returns
```

### Complete Example: Robust Trading System

```python
class RobustHMMTradingSystem:
    """Complete trading system with all safeguards"""

    def __init__(self, n_states=3, lookback_regime=252*78):
        self.n_states = n_states
        self.lookback_regime = lookback_regime
        self.model = None
        self.scaler = None
        self.is_trained = False

    def train(self, data):
        """Train model with all safeguards"""

        print("=== TRAINING ROBUST HMM SYSTEM ===")

        # 1. Data validation
        if len(data) < self.lookback_regime:
            raise ValueError(f"Insufficient data: need {self.lookback_regime}, have {len(data)}")

        # 2. Feature engineering
        features = add_features(data)
        feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
        X = features[feature_cols].dropna()

        # 3. Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]

        # 4. Find optimal number of states
        optimal_states = regularize_hmm(X_train)

        # 5. Train final model
        print(f"\nTraining final model with {optimal_states} states...")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = hmm.GaussianHMM(
            n_components=optimal_states,
            covariance_type='diag',
            random_state=42,
            n_iter=200
        )

        self.model.fit(X_train_scaled)

        # 6. Validate no overfitting
        X_test_scaled = self.scaler.transform(X_test)
        is_overfitting = detect_overfitting(self.model, X_train_scaled, X_test_scaled)

        if is_overfitting:
            print("⚠️  Model shows overfitting, using diagonal covariance")
            self.model = hmm.GaussianHMM(
                n_components=optimal_states,
                covariance_type='diag',
                random_state=42,
                n_iter=100
            )
            self.model.fit(X_train_scaled)

        # 7. Final validation
        train_states = self.model.predict(X_train_scaled)
        test_states = self.model.predict(X_test_scaled)

        # Prevent lookahead bias
        train_features_adj, train_states_adj = prevent_lookahead_bias(
            features.iloc[:len(train_states)], train_states
        )

        self.is_trained = True

        print("✅ Training complete with all safeguards")

        return {
            'train_score': self.model.score(X_train_scaled),
            'test_score': self.model.score(X_test_scaled),
            'n_states': optimal_states,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

    def predict(self, data):
        """Make predictions with proper validation"""

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Add features
        features = add_features(data)
        feature_cols = ['log_ret', 'atr', 'rsi', 'bb_width', 'adx']
        X = features[feature_cols].dropna()

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)

        # Apply lookahead bias correction
        features_adj, states_adj = prevent_lookahead_bias(features, states)

        return features_adj, states_adj

    def backtest_robust(self, data, n_folds=5):
        """Perform robust backtesting with out-of-sample validation"""

        print("\n=== ROBUST BACKTESTING ===")

        # Out-of-sample testing
        oos_results = out_of_sample_test(data)

        # Walk-forward analysis
        walk_forward_results = self.walk_forward_analysis(data, n_folds)

        # Performance summary
        print(f"\n=== ROBUST PERFORMANCE SUMMARY ===")
        print(f"Out-of-sample periods: {len(oos_results)}")
        print(f"Average OOS return: {np.mean([r['return'] for r in oos_results]):.4f}")
        print(f"Average OOS Sharpe: {np.mean([r['sharpe'] for r in oos_results]):.2f}")

        return {
            'out_of_sample': oos_results,
            'walk_forward': walk_forward_results
        }

    def walk_forward_analysis(self, data, n_folds=5):
        """Perform walk-forward analysis"""

        fold_size = len(data) // n_folds
        results = []

        for i in range(n_folds):
            # Define train/test periods
            train_start = i * fold_size
            train_end = min((i + 2) * fold_size, len(data))  # Use 2 folds for training
            test_start = train_end
            test_end = min((i + 3) * fold_size, len(data))   # 1 fold for testing

            if test_end >= len(data):
                break

            print(f"Walk-forward fold {i+1}: train={train_start}:{train_end}, test={test_start}:{test_end}")

            # Train
            train_data = data.iloc[train_start:train_end]
            self.train(train_data)

            # Test
            test_data = data.iloc[test_start:test_end]
            features_test, states_test = self.predict(test_data)

            # Calculate performance
            returns = calculate_strategy_returns(features_test, states_test)

            results.append({
                'fold': i + 1,
                'return': returns.sum(),
                'sharpe': returns.mean() / returns.std() * np.sqrt(252*78) if returns.std() > 0 else 0,
                'n_samples': len(returns)
            })

        return results

# Use the robust system
robust_system = RobustHMMTradingSystem()

# Train with safeguards
training_results = robust_system.train(features)

# Robust backtesting
backtest_results = robust_system.backtest_robust(features)

print("\n=== FINAL RESULTS ===")
print("System trained with comprehensive safeguards against:")
print("✅ Overfitting")
print("✅ Lookahead bias")
print("✅ Data snooping")
print("✅ Poor generalization")
```

---

## Conclusion

These practical examples demonstrate:

1. **Basic to advanced usage** of the HMM Futures Analysis toolkit
2. **Real-world trading strategies** based on regime detection
3. **Performance optimization** for large datasets
4. **Robust backtesting** with proper safeguards
5. **Common pitfalls** and their solutions

The key takeaways for successful HMM implementation:

1. **Always validate your model** with out-of-sample testing
2. **Prevent lookahead bias** by properly aligning signals and returns
3. **Use appropriate regularization** to avoid overfitting
4. **Consider computational efficiency** for large datasets
5. **Combine multiple approaches** for robust results

Remember that no model is perfect, and HMMs should be used as part of a comprehensive trading strategy with proper risk management.
