Momentum Strategy
=================

A momentum-based trading strategy that leverages HMM regime detection for futures markets.

Overview
--------

Momentum strategies capitalize on the persistence of price movements. When combined with HMM regime detection, we can enhance momentum signals by identifying market states where momentum is most likely to persist or reverse.

Strategy Logic
--------------

The momentum strategy identifies strong trending regimes and uses multiple timeframe momentum indicators to generate trading signals:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   class MomentumStrategy:
       def __init__(self, lookback_periods=[5, 10, 20], volatility_threshold=0.02,
                    momentum_threshold=0.001, regime_filter=True):
           """
           Initialize momentum strategy with HMM regime filtering.

           Parameters:
           - lookback_periods: List of periods for momentum calculation
           - volatility_threshold: Minimum volatility for trading
           - momentum_threshold: Minimum momentum for signal generation
           - regime_filter: Whether to use HMM regime filtering
           """
           self.lookback_periods = lookback_periods
           self.volatility_threshold = volatility_threshold
           self.momentum_threshold = momentum_threshold
           self.regime_filter = regime_filter
           self.regime_performance = {}

       def calculate_momentum_signals(self, data):
           """Calculate momentum signals across multiple timeframes."""
           signals = pd.DataFrame(index=data.index)
           signals['price'] = data['close']
           signals['returns'] = data['close'].pct_change()
           signals['volatility'] = signals['returns'].rolling(20).std()

           # Calculate momentum for different periods
           for period in self.lookback_periods:
               signals[f'momentum_{period}'] = data['close'].pct_change(period)
               signals[f'momentum_ma_{period}'] = signals[f'momentum_{period}'].rolling(5).mean()

           # Composite momentum score
           momentum_cols = [f'momentum_{period}' for period in self.lookback_periods]
           signals['momentum_score'] = signals[momentum_cols].mean(axis=1)
           signals['momentum_strength'] = abs(signals['momentum_score'])

           # Momentum confirmation (multiple timeframes aligned)
           positive_momentum = sum(1 for period in self.lookback_periods
                                 if signals[f'momentum_{period}'] > 0)
           negative_momentum = sum(1 for period in self.lookback_periods
                                 if signals[f'momentum_{period}'] < 0)

           signals['momentum_consensus'] = np.where(
               positive_momentum >= len(self.lookback_periods) * 0.7, 1,
               np.where(negative_momentum >= len(self.lookback_periods) * 0.7, -1, 0)
           )

           return signals

       def filter_by_regime(self, signals, states, regime_analysis):
           """Filter signals based on HMM regime performance."""
           if not self.regime_filter:
               return signals.copy()

           filtered_signals = signals.copy()
           filtered_signals['regime'] = states

           # Analyze historical performance by regime
           for state in states['hmm_state'].unique():
               state_mask = states['hmm_state'] == state
               state_data = signals[state_mask]

               if len(state_data) > 20:  # Need sufficient data
                   # Calculate regime-specific performance metrics
                   regime_returns = state_data['returns']
                   regime_momentum = state_data['momentum_score']

                   # Momentum effectiveness in this regime
                   momentum_signals = (regime_momentum > self.momentum_threshold).astype(int)
                   momentum_performance = (momentum_signals.shift(1) * regime_returns).mean()

                   # Store regime performance
                   self.regime_performance[state] = {
                       'avg_return': regime_returns.mean(),
                       'volatility': regime_returns.std(),
                       'momentum_effectiveness': momentum_performance,
                       'tendency': 'bullish' if regime_returns.mean() > 0 else 'bearish'
                   }

                   # Filter signals based on regime characteristics
                   if momentum_performance < 0:  # Momentum doesn't work in this regime
                       filtered_signals.loc[state_mask, 'momentum_consensus'] = 0

           return filtered_signals

       def generate_signals(self, data, states=None, regime_analysis=None):
           """Generate trading signals combining momentum and regime analysis."""
           # Calculate base momentum signals
           signals = self.calculate_momentum_signals(data)

           # Apply regime filtering if states are provided
           if states is not None and regime_analysis is not None:
               signals = self.filter_by_regime(signals, states, regime_analysis)

           # Generate final trading signals
           conditions = [
               # Strong momentum with sufficient volatility
               (signals['momentum_consensus'] == 1) &
               (signals['volatility'] > self.volatility_threshold) &
               (signals['momentum_strength'] > self.momentum_threshold),

               # Strong negative momentum
               (signals['momentum_consensus'] == -1) &
               (signals['volatility'] > self.volatility_threshold) &
               (signals['momentum_strength'] > self.momentum_threshold)
           ]

           choices = [1, -1]  # Buy for positive momentum, sell for negative
           default = 0  # No position

           signals['signal'] = np.select(conditions, choices, default)

           # Additional signal confirmation
           signals['signal_strength'] = signals['momentum_strength'] * signals['volatility']

           # Risk management: reduce position size in high volatility
           volatility_adjustment = np.where(
               signals['volatility'] > 0.05,  # High volatility threshold
               0.5,  # Reduce position by 50%
               np.where(
                   signals['volatility'] < 0.01,  # Low volatility threshold
                   1.5,  # Increase position by 50%
                   1.0  # Normal position
               )
           )

           signals['adjusted_signal'] = signals['signal'] * volatility_adjustment

           return signals

       def calculate_performance_metrics(self, signals, initial_capital=100000):
           """Calculate comprehensive performance metrics."""
           # Calculate strategy returns
           signals['strategy_returns'] = signals['adjusted_signal'].shift(1) * signals['returns']
           signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
           signals['equity_curve'] = initial_capital * signals['cumulative_returns']

           # Performance metrics
           total_return = signals['cumulative_returns'].iloc[-1] - 1
           annual_return = signals['strategy_returns'].mean() * 252
           volatility = signals['strategy_returns'].std() * np.sqrt(252)
           sharpe_ratio = annual_return / volatility if volatility > 0 else 0

           # Maximum drawdown
           cumulative_max = signals['cumulative_returns'].expanding().max()
           drawdown = (signals['cumulative_returns'] - cumulative_max) / cumulative_max
           max_drawdown = drawdown.min()

           # Win rate and profit factor
           win_rate = (signals['strategy_returns'] > 0).mean()
           profit_factor = abs(signals[signals['strategy_returns'] > 0]['strategy_returns'].sum() /
                             signals[signals['strategy_returns'] < 0]['strategy_returns'].sum()) if (signals['strategy_returns'] < 0).sum() != 0 else np.inf

           metrics = {
               'total_return': total_return,
               'annual_return': annual_return,
               'volatility': volatility,
               'sharpe_ratio': sharpe_ratio,
               'max_drawdown': max_drawdown,
               'win_rate': win_rate,
               'profit_factor': profit_factor,
               'total_trades': (signals['signal'] != 0).sum(),
               'final_equity': signals['equity_curve'].iloc[-1]
           }

           return signals, metrics

Implementation Example
---------------------

Complete example with backtesting:

.. code-block:: python

   def run_momentum_strategy_backtest(data, hmm_states, train_split=0.7):
       """Run complete momentum strategy backtest with HMM filtering."""

       # Split data for training and testing
       split_idx = int(len(data) * train_split)
       train_data = data.iloc[:split_idx]
       test_data = data.iloc[split_idx:]
       train_states = hmm_states.iloc[:split_idx]
       test_states = hmm_states.iloc[split_idx:]

       print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
       print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")

       # Initialize strategy
       strategy = MomentumStrategy(
           lookback_periods=[5, 10, 20],
           volatility_threshold=0.015,
           momentum_threshold=0.002,
           regime_filter=True
       )

       # Analyze regime performance on training data
       print("\nAnalyzing regime performance on training data...")
       train_signals = strategy.calculate_momentum_signals(train_data)

       # Generate signals for test period
       print("Generating signals for test period...")
       test_signals = strategy.generate_signals(test_data, test_states, None)

       # Calculate performance
       test_signals, performance = strategy.calculate_performance_metrics(test_signals)

       # Display results
       print("\n" + "="*50)
       print("MOMENTUM STRATEGY PERFORMANCE")
       print("="*50)

       for metric, value in performance.items():
           if metric in ['total_return', 'annual_return', 'max_drawdown', 'win_rate', 'profit_factor']:
               print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
           elif metric in ['volatility', 'sharpe_ratio']:
               print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
           else:
               print(f"{metric.replace('_', ' ').title()}: {value}")

       # Analyze performance by regime
       print("\n" + "="*50)
       print("PERFORMANCE BY HMM REGIME")
       print("="*50)

       test_signals['regime'] = test_states['hmm_state']
       regime_performance = {}

       for state in test_states['hmm_state'].unique():
           state_data = test_signals[test_signals['regime'] == state]
           if len(state_data) > 10:
               state_returns = state_data['strategy_returns']
               regime_performance[state] = {
                   'periods': len(state_data),
                   'avg_return': state_returns.mean(),
                   'volatility': state_returns.std(),
                   'sharpe': state_returns.mean() / state_returns.std() * np.sqrt(252) if state_returns.std() > 0 else 0,
                   'win_rate': (state_returns > 0).mean(),
                   'total_return': (1 + state_returns).cumprod().iloc[-1] - 1
               }

               print(f"\nState {state}:")
               for metric, value in regime_performance[state].items():
                   if isinstance(value, float):
                       print(f"  {metric}: {value:.4f}")
                   else:
                       print(f"  {metric}: {value}")

       return test_signals, performance, regime_performance

   # Run the backtest
   signals, performance, regime_perf = run_momentum_strategy_backtest(results, results)

Advanced Features
-----------------

Dynamic Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adapt strategy parameters based on market conditions:

.. code-block:: python

   class AdaptiveMomentumStrategy(MomentumStrategy):
       def __init__(self, base_lookback_periods=[5, 10, 20], **kwargs):
           super().__init__(base_lookback_periods, **kwargs)
           self.base_lookback_periods = base_lookback_periods
           self.market_state = 'normal'

       def detect_market_state(self, data):
           """Detect current market state for parameter adjustment."""
           returns = data['close'].pct_change()
           volatility = returns.rolling(20).std().iloc[-1]
           trend_strength = abs(returns.rolling(50).mean().iloc[-1])

           # Classify market state
           if volatility > 0.04:
               self.market_state = 'high_volatility'
           elif volatility < 0.01:
               self.market_state = 'low_volatility'
           elif trend_strength > 0.002:
               self.market_state = 'strong_trend'
           else:
               self.market_state = 'normal'

           return self.market_state

       def get_adapted_parameters(self):
           """Get strategy parameters adapted to current market state."""
           params = {
               'normal': {
                   'lookback_periods': [5, 10, 20],
                   'volatility_threshold': 0.015,
                   'momentum_threshold': 0.002
               },
               'high_volatility': {
                   'lookback_periods': [3, 8, 15],  # Shorter periods
                   'volatility_threshold': 0.025,  # Higher threshold
                   'momentum_threshold': 0.003   # Stronger signal required
               },
               'low_volatility': {
                   'lookback_periods': [8, 15, 30],  # Longer periods
                   'volatility_threshold': 0.010,  # Lower threshold
                   'momentum_threshold': 0.001   # Weaker signal acceptable
               },
               'strong_trend': {
                   'lookback_periods': [5, 15, 25],  # Emphasize medium-term
                   'volatility_threshold': 0.012,
                   'momentum_threshold': 0.0015
               }
           }

           return params.get(self.market_state, params['normal'])

       def generate_adaptive_signals(self, data, states=None):
           """Generate signals with adaptive parameters."""
           # Detect market state
           self.detect_market_state(data)

           # Get adapted parameters
           adapted_params = self.get_adapted_parameters()

           # Update strategy parameters
           self.lookback_periods = adapted_params['lookback_periods']
           self.volatility_threshold = adapted_params['volatility_threshold']
           self.momentum_threshold = adapted_params['momentum_threshold']

           # Generate signals with adapted parameters
           return self.generate_signals(data, states)

Multi-Asset Momentum
~~~~~~~~~~~~~~~~~~~~

Extend momentum strategy across multiple futures contracts:

.. code-block:: python

   class MultiAssetMomentumStrategy:
       def __init__(self, assets, lookback_periods=[5, 10, 20]):
           self.assets = assets
           self.lookback_periods = lookback_periods
           self.asset_performance = {}

       def calculate_cross_asset_momentum(self, price_data):
           """Calculate momentum signals across multiple assets."""
           momentum_signals = {}

           for asset in self.assets:
               if asset in price_data.columns:
                   asset_data = price_data[asset].dropna()
                   returns = asset_data.pct_change()

                   # Calculate momentum for different periods
                   momentum_scores = []
                   for period in self.lookback_periods:
                       momentum = asset_data.pct_change(period)
                       momentum_scores.append(momentum)

                   # Composite momentum score
                   composite_momentum = pd.concat(momentum_scores, axis=1).mean(axis=1)
                   momentum_signals[asset] = composite_momentum

           return pd.DataFrame(momentum_signals)

       def calculate_relative_momentum(self, momentum_signals):
           """Calculate relative momentum between assets."""
           # Rank assets by momentum
           momentum_rank = momentum_signals.rank(axis=1, pct=True)

           # Calculate relative momentum (momentum relative to peers)
           relative_momentum = momentum_signals.sub(momentum_signals.mean(axis=1), axis=0)

           return momentum_rank, relative_momentum

       def generate_portfolio_signals(self, price_data, top_n=3, bottom_n=1):
           """Generate portfolio-level signals."""
           # Calculate momentum signals
           momentum_signals = self.calculate_cross_asset_momentum(price_data)
           momentum_rank, relative_momentum = self.calculate_relative_momentum(momentum_signals)

           # Generate portfolio weights
           portfolio_weights = pd.DataFrame(0, index=momentum_signals.index,
                                          columns=momentum_signals.columns)

           # Long top performers
           for i, row in momentum_rank.iterrows():
               if not row.isna().all():
                   top_assets = row.nlargest(top_n).index
                   bottom_assets = row.nsmallest(bottom_n).index

                   # Equal weight long positions
                   portfolio_weights.loc[i, top_assets] = 1.0 / top_n

                   # Short positions (optional)
                   if bottom_n > 0:
                       portfolio_weights.loc[i, bottom_assets] = -1.0 / bottom_n

           return portfolio_weights, momentum_signals

Risk Management
----------------

Position Sizing
~~~~~~~~~~~~~~~

Implement sophisticated position sizing based on volatility and regime:

.. code-block:: python

   def calculate_position_size(signals, base_size=0.1, max_size=0.3):
       """Calculate dynamic position sizes based on risk metrics."""

       # Volatility-based position sizing
       vol_target = 0.15  # Target annual volatility
       current_vol = signals['volatility'] * np.sqrt(252)
       vol_adjusted_size = base_size * (vol_target / current_vol)

       # Regime-based adjustments
       regime_adjustments = {}
       for state in signals['regime'].unique():
           state_data = signals[signals['regime'] == state]
           if len(state_data) > 20:
               # Adjust based on historical performance
               state_sharpe = (state_data['returns'].mean() / state_data['returns'].std() * np.sqrt(252))
               regime_adjustments[state] = min(max(0.5, state_sharpe / 1.0), 1.5)
           else:
               regime_adjustments[state] = 1.0

       # Apply regime adjustments
       regime_adj = signals['regime'].map(regime_adjustments)
       final_size = vol_adjusted_size * regime_adj

       # Apply size limits
       final_size = final_size.clip(lower=0.05, upper=max_size)

       return final_size

   def implement_stop_loss(signals, stop_loss_pct=0.03):
       """Implement stop-loss mechanisms."""
       signals_with_stop = signals.copy()

       # Calculate running P&L for positions
       signals_with_stop['position'] = signals_with_stop['signal'].shift(1)
       signals_with_stop['position_pnl'] = signals_with_stop['position'] * signals_with_stop['returns']

       # Calculate cumulative P&L for each trade
       trade_start = (signals_with_stop['position'] != signals_with_stop['position'].shift(1)).cumsum()
       signals_with_stop['trade_pnl'] = signals_with_stop.groupby(trade_start)['position_pnl'].cumsum()

       # Apply stop-loss
       stop_loss_signal = np.where(
           (signals_with_stop['position'] != 0) &
           (signals_with_stop['trade_pnl'] < -stop_loss_pct),
           -signals_with_stop['position'],  # Exit position
           0
       )

       signals_with_stop['stop_loss_signal'] = stop_loss_signal

       # Combine original signals with stop-loss
       final_signal = signals_with_stop['signal'].copy()
       final_signal[signals_with_stop['stop_loss_signal'] != 0] = signals_with_stop['stop_loss_signal']

       signals_with_stop['final_signal'] = final_signal

       return signals_with_stop

Performance Attribution
-----------------------

Strategy Decomposition
~~~~~~~~~~~~~~~~~~~~~~

Analyze which components of the strategy are driving performance:

.. code-block:: python

   def analyze_momentum_performance_decomposition(signals):
       """Decompose strategy performance by component."""

       attribution = pd.DataFrame(index=signals.index)

       # Base momentum performance
       attribution['base_momentum'] = signals['momentum_consensus'].shift(1) * signals['returns']

       # Volatility filtering contribution
       vol_filter = signals['volatility'] > signals['volatility'].quantile(0.3)
       attribution['volatility_filter'] = (
           (signals['momentum_consensus'] * vol_filter).shift(1) * signals['returns']
       )

       # Regime filtering contribution
       if 'regime' in signals.columns:
           regime_filter = signals['regime'].isin([s for s, perf in
                                                 strategy.regime_performance.items()
                                                 if perf.get('momentum_effectiveness', 0) > 0])
           attribution['regime_filter'] = (
               (signals['momentum_consensus'] * regime_filter).shift(1) * signals['returns']
           )

       # Position sizing contribution
       attribution['position_sizing'] = signals['adjusted_signal'].shift(1) * signals['returns']

       # Calculate component contributions
       components = {}
       for col in attribution.columns:
           if col != 'position_sizing':  # This is the final strategy
               total_return = attribution[col].sum()
               sharpe = (attribution[col].mean() / attribution[col].std() * np.sqrt(252)
                        if attribution[col].std() > 0 else 0)
               components[col] = {
                   'total_return': total_return,
                   'sharpe_ratio': sharpe,
                   'contribution_pct': (total_return / attribution['position_sizing'].sum() * 100
                                       if attribution['position_sizing'].sum() != 0 else 0)
               }

       # Display attribution results
       print("\n" + "="*50)
       print("PERFORMANCE ATTRIBUTION ANALYSIS")
       print("="*50)

       for component, metrics in components.items():
           print(f"\n{component.replace('_', ' ').title()}:")
           print(f"  Total Return: {metrics['total_return']:.4f}")
           print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
           print(f"  Contribution: {metrics['contribution_pct']:.1f}%")

       return attribution, components

   # Analyze performance attribution
   attribution, components = analyze_momentum_performance_decomposition(signals)

This momentum strategy demonstrates how HMM regime detection can enhance traditional momentum approaches by identifying market conditions where momentum is most likely to be effective and avoiding periods where momentum strategies typically underperform.