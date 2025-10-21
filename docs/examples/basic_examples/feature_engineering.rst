Feature Engineering
===================

Comprehensive guide to creating and using technical indicators for HMM analysis.

Overview
--------

Feature engineering is the process of creating meaningful predictors from raw OHLCV data that help Hidden Markov Models identify market regimes more effectively. Good features capture different aspects of market behavior:

- **Trend indicators**: Direction and strength of price movements
- **Momentum indicators**: Speed and change of price movements
- **Volatility indicators**: Market uncertainty and risk
- **Volume indicators**: Market participation and conviction
- **Pattern indicators**: Reversal and continuation patterns

Built-in Features
-----------------

The system includes a comprehensive set of pre-built technical indicators:

Price-Based Features
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.data_processing.feature_engineering import add_features

   # Basic returns
   - returns: Daily returns (close_t / close_{t-1} - 1)
   - log_returns: Logarithmic returns for better statistical properties
   - pct_change: Percentage change across multiple periods

   # Price levels
   - high_low_pct: High-low percentage range
   - close_pct: Position of close within daily range
   - typical_price: (high + low + close) / 3
   - weighted_price: (high + low + 2*close) / 4

Moving Average Features
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple moving averages
   - sma_5, sma_10, sma_20, sma_50: Short to long-term trends
   - ema_5, ema_10, ema_20: Exponential moving averages
   - price_sma_ratio: Current price relative to moving average
   - sma_crossover: Moving average crossover signals

   # Moving average relationships
   - sma_distance: Distance between short and long MAs
   - ma_slope: Slope of moving average (trend strength)

Volatility Features
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Volatility measures
   - volatility_5, volatility_14, volatility_30: Rolling standard deviation
   - atr_14: Average True Range (normalized)
   - vix_proxy: Volatility index approximation
   - vol_ratio: Short-term vs long-term volatility

   # Volatility regimes
   - vol_regime: Categorical volatility levels
   - vol_spike: Unusual volatility increases

Momentum Features
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Momentum indicators
   - rsi_14: Relative Strength Index (0-100)
   - momentum_5, momentum_10: Price momentum over periods
   - roc_5, roc_10: Rate of change
   - acceleration: Change in momentum

   # Oscillators
   - stoch_k, stoch_d: Stochastic oscillator
   - cci: Commodity Channel Index
   - williams_r: Williams %R

Volume Features
~~~~~~~~~~~~~~~

.. code-block:: python

   # Volume analysis
   - volume_sma_20: Volume moving average
   - volume_ratio: Current vs average volume
   - price_volume_trend: Price-volume correlation
   - on_balance_volume: Cumulative volume indicator

   # Volume patterns
   - volume_spike: Unusual volume increases
   - accumulation_distribution: Money flow indicator

Pattern Features
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Price patterns
   - doji: Doji candlestick pattern
   - hammer: Hammer candlestick pattern
   - engulfing: Engulfing patterns
   - gap: Price gaps from previous close

   # Trend patterns
   - higher_highs: Series of higher highs
   - lower_lows: Series of lower lows
   - consolidation: Range-bound markets

Custom Feature Engineering
--------------------------

Creating Custom Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can easily add your own technical indicators:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from src.data_processing.feature_engineering import FeatureEngineer

   def custom_momentum_indicator(data, period=10):
       """Custom momentum indicator based on price acceleration."""
       returns = data['close'].pct_change()
       momentum = returns.rolling(period).mean()
       acceleration = momentum.diff()
       return acceleration

   def custom_volatility_regime(data, window=20):
       """Classify volatility into regimes."""
       volatility = data['close'].pct_change().rolling(window).std()
       regime = pd.cut(volatility,
                      bins=[0, volatility.quantile(0.33),
                           volatility.quantile(0.67), np.inf],
                      labels=['Low', 'Medium', 'High'])
       return regime

   # Add custom features
   engineer = FeatureEngineer()
   engineer.add_feature('custom_momentum', custom_momentum_indicator)
   engineer.add_feature('vol_regime', custom_volatility_regime)

   features = engineer.process(data)

Domain-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~

For futures markets, consider these specialized features:

.. code-block:: python

   def roll_yield(data, contract_months=[3, 6, 9]):
       """Calculate roll yield between contract months."""
       # Implementation for calculating roll yields
       pass

   def term_structure(data, short_term=1, long_term=3):
       """Term structure of futures contracts."""
       # Implementation for term structure analysis
       pass

   def seasonality_features(data):
       """Extract seasonal patterns from futures data."""
       data = data.copy()
       data['month'] = pd.to_datetime(data['datetime']).dt.month
       data['quarter'] = pd.to_datetime(data['datetime']).dt.quarter
       data['day_of_week'] = pd.to_datetime(data['datetime']).dt.dayofweek

       # Seasonal dummies
       for month in range(1, 13):
           data[f'month_{month}'] = (data['month'] == month).astype(int)

       return data

Feature Selection
-----------------

Correlation Analysis
~~~~~~~~~~~~~~~~~~~

Remove highly correlated features to avoid redundancy:

.. code-block:: python

   import seaborn as sns
   import matplotlib.pyplot as plt

   def analyze_feature_correlation(features, threshold=0.9):
       """Analyze and remove highly correlated features."""
       # Calculate correlation matrix
       corr_matrix = features.corr().abs()

       # Find highly correlated pairs
       high_corr = np.where(corr_matrix > threshold)
       high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                   for x, y in zip(*high_corr) if x != y]

       # Remove duplicates
       high_corr = list(set(tuple(sorted(item[:2])) + (item[2],) for item in high_corr))

       print("Highly correlated feature pairs:")
       for feat1, feat2, corr in sorted(high_corr, key=lambda x: x[2], reverse=True):
           print(f"  {feat1} - {feat2}: {corr:.3f}")

       return high_corr

   # Analyze correlations
   correlated_features = analyze_feature_correlation(features)

Mutual Information
~~~~~~~~~~~~~~~~~~

Use mutual information to identify the most predictive features:

.. code-block:: python

   from sklearn.feature_selection import mutual_info_regression
   from sklearn.preprocessing import LabelEncoder

   def select_features_by_mutual_info(X, y, top_k=20):
       """Select top-k features based on mutual information."""
       # Handle categorical features
       X_encoded = X.copy()
       for col in X.select_dtypes(include=['object', 'category']).columns:
           le = LabelEncoder()
           X_encoded[col] = le.fit_transform(X[col].astype(str))

       # Calculate mutual information
       mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
       mi_df = pd.DataFrame({
           'feature': X.columns,
           'mi_score': mi_scores
       }).sort_values('mi_score', ascending=False)

       return mi_df.head(top_k)

   # Select features for HMM
   X = features.drop(['datetime'], axis=1, errors='ignore')
   y = features['returns']  # Use returns as target for MI analysis
   top_features = select_features_by_mutual_info(X, y, top_k=15)
   print("Top features by mutual information:")
   print(top_features)

Feature Engineering Pipeline
---------------------------

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def complete_feature_pipeline(data, target_col='returns'):
       """Complete feature engineering pipeline."""
       import pandas as pd
       import numpy as np
       from sklearn.preprocessing import StandardScaler
       from sklearn.feature_selection import SelectKBest, f_regression

       # Step 1: Basic feature engineering
       features = add_features(data)

       # Step 2: Remove features with too many missing values
       missing_threshold = 0.1
       features = features.dropna(thresh=len(features) * (1 - missing_threshold), axis=1)

       # Step 3: Handle remaining missing values
       numeric_features = features.select_dtypes(include=[np.number])
       features[numeric_features.columns] = numeric_features.fillna(numeric_features.mean())

       # Step 4: Remove highly correlated features
       corr_matrix = features.corr().abs().values
       upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

       # Find and remove correlated features
       to_remove = set()
       for i in range(corr_matrix.shape[0]):
           for j in range(i+1, corr_matrix.shape[1]):
               if upper_tri[i, j] and corr_matrix[i, j] > 0.9:
                   to_remove.add(features.columns[j])

       features = features.drop(columns=list(to_remove))

       # Step 5: Feature scaling
       scaler = StandardScaler()
       numeric_cols = features.select_dtypes(include=[np.number]).columns
       features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

       # Step 6: Feature selection
       if target_col in features.columns:
           X = features.drop(target_col, axis=1)
           y = features[target_col]

           # Remove non-numeric columns for feature selection
           X_numeric = X.select_dtypes(include=[np.number])

           selector = SelectKBest(f_regression, k=min(20, len(X_numeric.columns)))
           X_selected = selector.fit_transform(X_numeric, y)

           selected_features = X_numeric.columns[selector.get_support()]
           features = features[list(selected_features) + [target_col]]

       return features

   # Apply complete pipeline
   engineered_features = complete_feature_pipeline(data)
   print(f"Final feature set: {engineered_features.shape[1]} features")
   print(f"Features: {list(engineered_features.columns)}")

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

.. code-block:: python

   def validate_features(features):
       """Validate engineered features for quality issues."""
       issues = []

       # Check for infinite values
       inf_values = np.isinf(features.select_dtypes(include=[np.number])).sum()
       if inf_values.sum() > 0:
           issues.append(f"Infinite values found: {inf_values.sum()}")

       # Check for extreme outliers
       numeric_cols = features.select_dtypes(include=[np.number]).columns
       for col in numeric_cols:
           q1, q3 = features[col].quantile([0.25, 0.75])
           iqr = q3 - q1
           outliers = ((features[col] < q1 - 3*iqr) | (features[col] > q3 + 3*iqr)).sum()
           if outliers > len(features) * 0.01:  # More than 1% extreme outliers
               issues.append(f"Extreme outliers in {col}: {outliers}")

       # Check for constant features
       constant_features = []
       for col in numeric_cols:
           if features[col].std() < 1e-8:
               constant_features.append(col)

       if constant_features:
           issues.append(f"Constant features: {constant_features}")

       if issues:
           print("Feature validation issues:")
           for issue in issues:
               print(f"  - {issue}")
       else:
           print("✅ All features passed validation")

       return issues

Regime-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~

Different market regimes may require different features:

.. code-block:: python

   def regime_specific_features(data, regimes=None):
       """Create features specific to market regimes."""
       if regimes is None:
           # Basic regime classification
           returns = data['close'].pct_change()
           volatility = returns.rolling(20).std()
           trend = returns.rolling(50).mean()

           regimes = pd.Series(index=data.index)
           regimes[(volatility < volatility.quantile(0.33)) &
                   (abs(trend) < 0.001)] = 'Ranging'
           regimes[(volatility < volatility.quantile(0.33)) &
                   (trend > 0.001)] = 'Trending Up'
           regimes[(volatility < volatility.quantile(0.33)) &
                   (trend < -0.001)] = 'Trending Down'
           regimes[volatility >= volatility.quantile(0.33)] = 'Volatile'

       features = data.copy()
       features['regime'] = regimes

       # Regime-specific features
       for regime in regimes.unique():
           if pd.isna(regime):
               continue

           regime_mask = regimes == regime
           regime_returns = data.loc[regime_mask, 'close'].pct_change()

           features[f'regime_{regime}_volatility'] = regime_returns.rolling(10).std()
           features[f'regime_{regime}_momentum'] = regime_returns.rolling(5).mean()
           features[f'regime_{regime}_duration'] = regime_mask.groupby(
               (regime_mask != regime_mask.shift()).cumsum()
           ).cumcount()

       return features

Advanced Techniques
------------------

Feature Crosses
~~~~~~~~~~~~~~~

Create interaction terms between features:

.. code-block:: python

   def create_feature_crosses(features, cross_pairs=None):
       """Create interaction terms between feature pairs."""
       if cross_pairs is None:
           # Automatically find good pairs based on correlation
           numeric_features = features.select_dtypes(include=[np.number])
           corr_matrix = numeric_features.corr()

           # Find moderately correlated pairs (0.3 to 0.7)
           cross_pairs = []
           for i in range(len(corr_matrix.columns)):
               for j in range(i+1, len(corr_matrix.columns)):
                   corr = abs(corr_matrix.iloc[i, j])
                   if 0.3 <= corr <= 0.7:
                       cross_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

       crossed_features = features.copy()

       for feat1, feat2 in cross_pairs:
           if feat1 in features.columns and feat2 in features.columns:
               cross_name = f"{feat1}_x_{feat2}"
               crossed_features[cross_name] = features[feat1] * features[feat2]

       return crossed_features

Temporal Features
~~~~~~~~~~~~~~~~~

Add time-based features for futures markets:

.. code-block:: python

   def add_temporal_features(data):
       """Add temporal features for futures analysis."""
       features = data.copy()

       # Convert datetime if needed
       if 'datetime' in features.columns:
           features['datetime'] = pd.to_datetime(features['datetime'])

           # Time-based features
           features['hour'] = features['datetime'].dt.hour
           features['day_of_week'] = features['datetime'].dt.dayofweek
           features['month'] = features['datetime'].dt.month
           features['quarter'] = features['datetime'].dt.quarter
           features['day_of_year'] = features['datetime'].dt.dayofyear

           # Cyclical encoding
           features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
           features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
           features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
           features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

           # Futures-specific temporal features
           features['is_month_end'] = features['datetime'].dt.is_month_end.astype(int)
           features['is_quarter_end'] = features['datetime'].dt.is_quarter_end.astype(int)

           # Trading session features (if intraday)
           features['is_us_session'] = ((features['hour'] >= 9) &
                                       (features['hour'] <= 16)).astype(int)
           features['is_asian_session'] = ((features['hour'] >= 19) |
                                          (features['hour'] <= 2)).astype(int)

       return features

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Missing Values After Rolling Windows:**

.. code-block:: python

   # Handle missing values from rolling calculations
   def handle_rolling_nan(features, method='forward_fill'):
       """Handle NaN values from rolling window calculations."""
       if method == 'forward_fill':
           return features.fillna(method='ffill').fillna(method='bfill')
       elif method == 'interpolate':
           return features.interpolate()
       elif method == 'drop':
           return features.dropna()
       else:
           raise ValueError(f"Unknown method: {method}")

**Numerical Stability:**

.. code-block:: python

   # Add small constants to avoid division by zero
   def ensure_numerical_stability(features, epsilon=1e-8):
       """Add small constants to ensure numerical stability."""
       features = features.copy()

       # Add epsilon to denominators
       for col in features.columns:
           if 'ratio' in col.lower() or 'pct' in col.lower():
               features[col] = features[col].replace([np.inf, -np.inf], np.nan)
               features[col] = features[col].fillna(0)

       return features

**Memory Usage:**

.. code-block:: python

   # Optimize memory usage for large datasets
   def optimize_memory(features):
       """Optimize memory usage by downcasting data types."""
       for col in features.columns:
           if features[col].dtype == 'float64':
               features[col] = pd.to_numeric(features[col], downcast='float')
           elif features[col].dtype == 'int64':
               features[col] = pd.to_numeric(features[col], downcast='integer')

       return features

This comprehensive feature engineering guide provides the foundation for creating high-quality features that improve HMM regime detection performance in futures markets.