Basic Visualization
===================

Creating effective visualizations to understand HMM results and market regimes.

Overview
--------

Visualization is crucial for interpreting HMM results and communicating findings. This guide covers creating professional charts and plots that help understand market regimes, state transitions, and model performance.

Setup and Dependencies
----------------------

Required Libraries
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Install required visualization libraries
   !uv install matplotlib seaborn plotly mplfinance dash

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   import plotly.graph_objects as go
   import plotly.express as px
   from plotly.subplots import make_subplots
   import mplfinance as mpf

   # Set style and configuration
   plt.style.use('seaborn-v0_8-darkgrid')
   sns.set_palette("husl")
   plt.rcParams['figure.figsize'] = (15, 10)
   plt.rcParams['font.size'] = 12

Price Charts with States
------------------------

Basic Price Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

The most fundamental visualization is showing price data colored by HMM states:

.. code-block:: python

   def plot_price_with_states(data, states, title="Price Chart with HMM States"):
       """Plot price data with HMM states as background colors."""

       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
       fig.suptitle(title, fontsize=16, fontweight='bold')

       # Get unique states and assign colors
       unique_states = sorted(states['hmm_state'].unique())
       colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
       state_colors = dict(zip(unique_states, colors))

       # Plot 1: Price with state backgrounds
       ax1.set_title('Price with Market Regimes')
       ax1.set_ylabel('Price ($)')

       # Plot price line
       ax1.plot(data['datetime'], data['close'], color='black', linewidth=1.5, alpha=0.8, label='Price')

       # Add state-colored backgrounds
       for i in range(len(states)):
           state = states['hmm_state'].iloc[i]
           start_date = states['datetime'].iloc[i]

           # Find the end of this state period
           end_idx = i
           while end_idx < len(states) - 1 and states['hmm_state'].iloc[end_idx + 1] == state:
               end_idx += 1

           end_date = states['datetime'].iloc[end_idx]

           # Add colored background
           ax1.axvspan(start_date, end_date, alpha=0.3, color=state_colors[state],
                      label=f'State {state}' if i == 0 or states['hmm_state'].iloc[i-1] != state else "")

           i = end_idx

       ax1.legend(loc='upper left')
       ax1.grid(True, alpha=0.3)

       # Plot 2: Volume with state colors
       ax2.set_title('Volume by Market Regime')
       ax2.set_xlabel('Date')
       ax2.set_ylabel('Volume')

       for state in unique_states:
           state_mask = states['hmm_state'] == state
           ax2.bar(states.loc[state_mask, 'datetime'],
                  states.loc[state_mask, 'volume'],
                  alpha=0.7, color=state_colors[state], label=f'State {state}')

       ax2.legend(loc='upper left')
       ax2.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()

   # Create basic price visualization
   plot_price_with_states(results, results, "Futures Price with HMM Market Regimes")

Candlestick Charts with States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For financial analysis, candlestick charts are more informative:

.. code-block:: python

   def plot_candlestick_with_states(data, states, title="Candlestick Chart with HMM States"):
       """Create a candlestick chart with HMM state backgrounds."""

       # Prepare data for mplfinance
       ohlcv_data = data.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
       ohlcv_data.index = pd.to_datetime(ohlcv_data.index)

       # Get unique states and colors
       unique_states = sorted(states['hmm_state'].unique())
       colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
       state_colors = dict(zip(unique_states, colors))

       # Create custom style
       style = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'font.size': 10})

       # Create addplot for volume
       volume_plot = mpf.make_addplot(ohlcv_data['volume'], panel=1, color='lightblue', alpha=0.7)

       # Plot with state backgrounds (need to do this manually)
       fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
       fig.suptitle(title, fontsize=16, fontweight='bold')

       # Plot candlesticks
       mpf.plot(ohlcv_data, type='candle', style=style, ax=axes[0], volume=False, show_nontrading=False)

       # Add state backgrounds to price chart
       current_state = None
       start_idx = None

       for i, (idx, row) in enumerate(ohlcv_data.iterrows()):
           state = states.loc[states['datetime'] == idx, 'hmm_state'].iloc[0]

           if current_state is None:
               current_state = state
               start_idx = i
           elif state != current_state:
               # End of current state period
               end_idx = i - 1
               start_date = ohlcv_data.index[start_idx]
               end_date = ohlcv_data.index[end_idx]

               axes[0].axvspan(start_date, end_date, alpha=0.3, color=state_colors[current_state])

               current_state = state
               start_idx = i

       # Add final state period
       if current_state is not None and start_idx is not None:
           start_date = ohlcv_data.index[start_idx]
           end_date = ohlcv_data.index[-1]
           axes[0].axvspan(start_date, end_date, alpha=0.3, color=state_colors[current_state])

       # Plot volume
       axes[1].bar(ohlcv_data.index, ohlcv_data['volume'], color='lightblue', alpha=0.7)
       axes[1].set_title('Volume')
       axes[1].set_ylabel('Volume')

       # Create legend for states
       legend_elements = [plt.Rectangle((0, 0), 1, 1, alpha=0.3, color=state_colors[state],
                                       label=f'State {state}') for state in unique_states]
       axes[0].legend(handles=legend_elements, loc='upper left')

       plt.tight_layout()
       plt.show()

   # Create candlestick visualization
   plot_candlestick_with_states(results, results, "Candlestick Chart with Market Regimes")

Interactive Visualizations
--------------------------

Plotly Interactive Charts
~~~~~~~~~~~~~~~~~~~~~~~~

Create interactive charts that allow zooming, hovering, and detailed inspection:

.. code-block:: python

   def create_interactive_hmm_chart(data, states):
       """Create an interactive Plotly chart with HMM states."""

       # Get unique states and colors
       unique_states = sorted(states['hmm_state'].unique())
       colors = px.colors.qualitative.Set3[:len(unique_states)]

       # Create figure with subplots
       fig = make_subplots(
           rows=3, cols=1,
           shared_xaxes=True,
           vertical_spacing=0.03,
           subplot_titles=('Price with HMM States', 'Volume', 'Returns'),
           row_width=[0.2, 0.2, 0.7]
       )

       # Plot 1: Price with state backgrounds
       fig.add_trace(
           go.Scatter(
               x=data['datetime'],
               y=data['close'],
               mode='lines',
               name='Price',
               line=dict(color='black', width=2)
           ),
           row=1, col=1
       )

       # Add state backgrounds as filled areas
       for i, state in enumerate(unique_states):
           state_mask = states['hmm_state'] == state
           state_data = states[state_mask]

           if len(state_data) > 0:
               # Create filled area for this state
               fig.add_trace(
                   go.Scatter(
                       x=state_data['datetime'].tolist() + state_data['datetime'].tolist()[-1:],
                       y=[data['close'].min()] * len(state_data) + [data['close'].max()],
                       mode='lines',
                       fill='toself',
                       fillcolor=colors[i],
                       opacity=0.3,
                       name=f'State {state}',
                       hoverinfo='skip'
                   ),
                   row=1, col=1
               )

       # Plot 2: Volume by state
       for i, state in enumerate(unique_states):
           state_mask = states['hmm_state'] == state
           state_data = states[state_mask]

           if len(state_data) > 0:
               fig.add_trace(
                   go.Bar(
                       x=state_data['datetime'],
                       y=state_data['volume'],
                       name=f'Volume State {state}',
                       marker_color=colors[i],
                       opacity=0.7
                   ),
                   row=2, col=1
               )

       # Plot 3: Returns by state
       for i, state in enumerate(unique_states):
           state_mask = states['hmm_state'] == state
           state_data = states[state_mask]

           if len(state_data) > 0:
               fig.add_trace(
                   go.Scatter(
                       x=state_data['datetime'],
                       y=state_data['returns'],
                       mode='markers',
                       name=f'Returns State {state}',
                       marker=dict(color=colors[i], size=4, opacity=0.8)
                   ),
                   row=3, col=1
               )

       # Update layout
       fig.update_layout(
           title='Interactive HMM Analysis Dashboard',
           height=900,
           showlegend=True,
           hovermode='x unified'
       )

       fig.update_xaxes(title_text="Date", row=3, col=1)
       fig.update_yaxes(title_text="Price ($)", row=1, col=1)
       fig.update_yaxes(title_text="Volume", row=2, col=1)
       fig.update_yaxes(title_text="Returns", row=3, col=1)

       fig.show()

   # Create interactive chart
   create_interactive_hmm_chart(results, results)

State Distribution Visualizations
---------------------------------

State Transition Heatmap
~~~~~~~~~~~~~~~~~~~~~~~~

Visualize how states transition over time:

.. code-block:: python

   def plot_state_transition_heatmap(state_series, title="State Transition Heatmap"):
       """Create a heatmap showing state transitions over time."""

       # Calculate transition matrix
       n_states = len(state_series.unique())
       transition_matrix = np.zeros((n_states, n_states))

       for i in range(len(state_series) - 1):
           from_state = state_series.iloc[i]
           to_state = state_series.iloc[i + 1]
           transition_matrix[from_state, to_state] += 1

       # Normalize to get probabilities
       row_sums = transition_matrix.sum(axis=1, keepdims=True)
       transition_probs = np.divide(transition_matrix, row_sums,
                                    where=row_sums != 0, out=np.zeros_like(transition_matrix))

       # Create heatmap
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

       # Plot 1: Count matrix
       sns.heatmap(transition_matrix, annot=True, fmt='g', cmap='Blues',
                   xticklabels=[f'State {i}' for i in range(n_states)],
                   yticklabels=[f'State {i}' for i in range(n_states)],
                   ax=ax1)
       ax1.set_title('Transition Count Matrix')
       ax1.set_xlabel('To State')
       ax1.set_ylabel('From State')

       # Plot 2: Probability matrix
       sns.heatmap(transition_probs, annot=True, fmt='.3f', cmap='Reds',
                   xticklabels=[f'State {i}' for i in range(n_states)],
                   yticklabels=[f'State {i}' for i in range(n_states)],
                   ax=ax2)
       ax2.set_title('Transition Probability Matrix')
       ax2.set_xlabel('To State')
       ax2.set_ylabel('From State')

       plt.suptitle(title, fontsize=16, fontweight='bold')
       plt.tight_layout()
       plt.show()

   # Plot transition heatmap
   plot_state_transition_heatmap(results['hmm_state'], "HMM State Transition Analysis")

State Duration Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Analyze how long states typically persist:

.. code-block:: python

   def plot_state_durations(state_series, title="State Duration Analysis"):
       """Analyze and visualize state duration patterns."""

       # Calculate state durations
       durations = []
       current_state = state_series.iloc[0]
       current_duration = 1

       for i in range(1, len(state_series)):
           if state_series.iloc[i] == current_state:
               current_duration += 1
           else:
               durations.append({'state': current_state, 'duration': current_duration})
               current_state = state_series.iloc[i]
               current_duration = 1

       # Add final duration
       durations.append({'state': current_state, 'duration': current_duration})

       durations_df = pd.DataFrame(durations)

       # Create visualization
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       fig.suptitle(title, fontsize=16, fontweight='bold')

       # Plot 1: Duration distribution by state
       ax1 = axes[0, 0]
       for state in durations_df['state'].unique():
           state_durations = durations_df[durations_df['state'] == state]['duration']
           ax1.hist(state_durations, alpha=0.7, label=f'State {state}', bins=20)

       ax1.set_title('State Duration Distribution')
       ax1.set_xlabel('Duration (periods)')
       ax1.set_ylabel('Frequency')
       ax1.legend()
       ax1.grid(True, alpha=0.3)

       # Plot 2: Box plot of durations
       ax2 = axes[0, 1]
       duration_data = [durations_df[durations_df['state'] == state]['duration']
                        for state in sorted(durations_df['state'].unique())]
       box_plot = ax2.boxplot(duration_data, labels=[f'State {s}' for s in sorted(durations_df['state'].unique())])
       ax2.set_title('State Duration Box Plot')
       ax2.set_xlabel('State')
       ax2.set_ylabel('Duration (periods)')
       ax2.grid(True, alpha=0.3)

       # Plot 3: State persistence over time
       ax3 = axes[1, 0]
       time_index = range(len(state_series))
       ax3.plot(time_index, state_series, linewidth=1, alpha=0.8)
       ax3.set_title('State Sequence Over Time')
       ax3.set_xlabel('Time')
       ax3.set_ylabel('State')
       ax3.set_yticks(sorted(state_series.unique()))
       ax3.grid(True, alpha=0.3)

       # Plot 4: Cumulative duration
       ax4 = axes[1, 1]
       cumulative_duration = durations_df.groupby('state')['duration'].cumsum()
       for state in durations_df['state'].unique():
           state_mask = durations_df['state'] == state
           ax4.plot(np.arange(len(cumulative_duration[state_mask])),
                   cumulative_duration[state_mask],
                   marker='o', label=f'State {state}')

       ax4.set_title('Cumulative Duration by State')
       ax4.set_xlabel('Transition Number')
       ax4.set_ylabel('Cumulative Duration')
       ax4.legend()
       ax4.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()

       return durations_df

   # Analyze state durations
   duration_analysis = plot_state_durations(results['hmm_state'])

Performance Visualizations
--------------------------

Return Distribution by State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare return characteristics across states:

.. code-block:: python

   def plot_return_distributions_by_state(data, states, title="Return Distributions by State"):
       """Visualize return distributions for each HMM state."""

       unique_states = sorted(states['hmm_state'].unique())
       n_states = len(unique_states)

       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       fig.suptitle(title, fontsize=16, fontweight='bold')

       # Plot 1: Histogram of returns by state
       ax1 = axes[0, 0]
       for state in unique_states:
           state_returns = data.loc[states['hmm_state'] == state, 'returns']
           if len(state_returns) > 0:
               ax1.hist(state_returns, bins=30, alpha=0.6,
                       label=f'State {state}', density=True)

       ax1.set_title('Return Distribution by State')
       ax1.set_xlabel('Returns')
       ax1.set_ylabel('Density')
       ax1.legend()
       ax1.grid(True, alpha=0.3)

       # Plot 2: Box plot of returns
       ax2 = axes[0, 1]
       return_data = []
       return_labels = []

       for state in unique_states:
           state_returns = data.loc[states['hmm_state'] == state, 'returns']
           if len(state_returns) > 0:
               return_data.append(state_returns)
               return_labels.append(f'State {state}')

       box_plot = ax2.boxplot(return_data, labels=return_labels)
       ax2.set_title('Return Distribution Box Plot')
       ax2.set_xlabel('State')
       ax2.set_ylabel('Returns')
       ax2.grid(True, alpha=0.3)

       # Plot 3: Cumulative returns by state
       ax3 = axes[1, 0]
       for state in unique_states:
           state_mask = states['hmm_state'] == state
           state_data = data[state_mask]
           if len(state_data) > 0:
               cumulative_returns = (1 + state_data['returns']).cumprod() - 1
               ax3.plot(state_data['datetime'], cumulative_returns,
                       label=f'State {state}', alpha=0.8)

       ax3.set_title('Cumulative Returns by State')
       ax3.set_xlabel('Date')
       ax3.set_ylabel('Cumulative Returns')
       ax3.legend()
       ax3.grid(True, alpha=0.3)

       # Plot 4: Risk-return scatter
       ax4 = axes[1, 1]
       risk_return_data = []

       for state in unique_states:
           state_returns = data.loc[states['hmm_state'] == state, 'returns']
           if len(state_returns) > 0:
               risk_return_data.append({
                   'state': state,
                   'mean_return': state_returns.mean(),
                   'volatility': state_returns.std(),
                   'sharpe': state_returns.mean() / state_returns.std() * np.sqrt(252) if state_returns.std() > 0 else 0
               })

       risk_return_df = pd.DataFrame(risk_return_data)

       scatter = ax4.scatter(risk_return_df['volatility'], risk_return_df['mean_return'],
                            s=100, alpha=0.7, c=risk_return_df['state'], cmap='viridis')

       ax4.set_title('Risk-Return Profile by State')
       ax4.set_xlabel('Volatility (Risk)')
       ax4.set_ylabel('Mean Return')

       # Add state labels
       for _, row in risk_return_df.iterrows():
           ax4.annotate(f"State {row['state']}",
                       (row['volatility'], row['mean_return']),
                       xytext=(5, 5), textcoords='offset points')

       ax4.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()

       return risk_return_df

   # Plot return distributions
   performance_analysis = plot_return_distributions_by_state(results, results)

Volatility Analysis
~~~~~~~~~~~~~~~~~~~

Compare volatility patterns across states:

.. code-block:: python

   def plot_volatility_analysis(data, states, title="Volatility Analysis by State"):
       """Analyze volatility patterns for each HMM state."""

       unique_states = sorted(states['hmm_state'].unique())

       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       fig.suptitle(title, fontsize=16, fontweight='bold')

       # Plot 1: Volatility time series by state
       ax1 = axes[0, 0]
       for state in unique_states:
           state_mask = states['hmm_state'] == state
           state_data = data[state_mask]
           if len(state_data) > 0:
               ax1.scatter(state_data['datetime'], state_data['volatility_14'],
                          label=f'State {state}', alpha=0.7, s=20)

       ax1.set_title('Volatility by State Over Time')
       ax1.set_xlabel('Date')
       ax1.set_ylabel('Volatility')
       ax1.legend()
       ax1.grid(True, alpha=0.3)

       # Plot 2: Volatility distribution
       ax2 = axes[0, 1]
       for state in unique_states:
           state_vol = data.loc[states['hmm_state'] == state, 'volatility_14']
           if len(state_vol) > 0:
               ax2.hist(state_vol, bins=20, alpha=0.6,
                       label=f'State {state}', density=True)

       ax2.set_title('Volatility Distribution by State')
       ax2.set_xlabel('Volatility')
       ax2.set_ylabel('Density')
       ax2.legend()
       ax2.grid(True, alpha=0.3)

       # Plot 3: Volatility vs Returns
       ax3 = axes[1, 0]
       for state in unique_states:
           state_mask = states['hmm_state'] == state
           state_data = data[state_mask]
           if len(state_data) > 0:
               ax3.scatter(state_data['volatility_14'], state_data['returns'],
                          label=f'State {state}', alpha=0.6, s=30)

       ax3.set_title('Volatility vs Returns by State')
       ax3.set_xlabel('Volatility')
       ax3.set_ylabel('Returns')
       ax3.legend()
       ax3.grid(True, alpha=0.3)

       # Plot 4: Volatility statistics
       ax4 = axes[1, 1]
       vol_stats = []
       state_labels = []

       for state in unique_states:
           state_vol = data.loc[states['hmm_state'] == state, 'volatility_14']
           if len(state_vol) > 0:
               vol_stats.append([state_vol.mean(), state_vol.median(), state_vol.std()])
               state_labels.append(f'State {state}')

       vol_stats = np.array(vol_stats)
       x = np.arange(len(state_labels))
       width = 0.25

       ax4.bar(x - width, vol_stats[:, 0], width, label='Mean', alpha=0.7)
       ax4.bar(x, vol_stats[:, 1], width, label='Median', alpha=0.7)
       ax4.bar(x + width, vol_stats[:, 2], width, label='Std Dev', alpha=0.7)

       ax4.set_title('Volatility Statistics by State')
       ax4.set_xlabel('State')
       ax4.set_ylabel('Volatility')
       ax4.set_xticks(x)
       ax4.set_xticklabels(state_labels)
       ax4.legend()
       ax4.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()

   # Plot volatility analysis
   plot_volatility_analysis(results, results)

Dashboard Creation
------------------

Comprehensive Dashboard
~~~~~~~~~~~~~~~~~~~~~~~

Create a comprehensive dashboard with multiple visualizations:

.. code-block:: python

   def create_hmm_dashboard(data, states, title="HMM Analysis Dashboard"):
       """Create a comprehensive dashboard for HMM analysis."""

       fig = plt.figure(figsize=(20, 16))
       fig.suptitle(title, fontsize=20, fontweight='bold')

       # Create grid specification
       gs = fig.add_gridspec(4, 3, height_ratios=[2, 1.5, 1.5, 1.5], width_ratios=[2, 2, 1])

       # Plot 1: Main price chart with states (spans 2 columns)
       ax1 = fig.add_subplot(gs[0, :2])

       unique_states = sorted(states['hmm_state'].unique())
       colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
       state_colors = dict(zip(unique_states, colors))

       # Plot price line
       ax1.plot(data['datetime'], data['close'], color='black', linewidth=2, alpha=0.8)

       # Add state backgrounds
       current_state = None
       start_idx = None

       for i in range(len(states)):
           state = states['hmm_state'].iloc[i]
           if current_state is None:
               current_state = state
               start_idx = i
           elif state != current_state:
               end_idx = i - 1
               ax1.axvspan(data['datetime'].iloc[start_idx], data['datetime'].iloc[end_idx],
                          alpha=0.3, color=state_colors[current_state])
               current_state = state
               start_idx = i

       # Add final state period
       if current_state is not None and start_idx is not None:
           ax1.axvspan(data['datetime'].iloc[start_idx], data['datetime'].iloc[-1],
                      alpha=0.3, color=state_colors[current_state])

       ax1.set_title('Price with Market Regimes', fontsize=14, fontweight='bold')
       ax1.set_ylabel('Price ($)')
       ax1.grid(True, alpha=0.3)

       # Create legend
       legend_elements = [plt.Rectangle((0, 0), 1, 1, alpha=0.3, color=state_colors[state],
                                       label=f'State {state}') for state in unique_states]
       ax1.legend(handles=legend_elements, loc='upper left')

       # Plot 2: State distribution pie chart
       ax2 = fig.add_subplot(gs[0, 2])
       state_counts = states['hmm_state'].value_counts().sort_index()
       colors_pie = [state_colors[state] for state in state_counts.index]
       ax2.pie(state_counts.values, labels=[f'State {i}' for i in state_counts.index],
              autopct='%1.1f%%', colors=colors_pie, startangle=90)
       ax2.set_title('State Distribution', fontsize=12, fontweight='bold')

       # Plot 3: Volume by state
       ax3 = fig.add_subplot(gs[1, 0])
       for state in unique_states:
           state_mask = states['hmm_state'] == state
           state_data = states[state_mask]
           if len(state_data) > 0:
               ax3.bar(state_data['datetime'], state_data['volume'],
                      alpha=0.7, color=state_colors[state], label=f'State {state}')

       ax3.set_title('Volume by State', fontsize=12)
       ax3.set_ylabel('Volume')
       ax3.grid(True, alpha=0.3)

       # Plot 4: Returns distribution
       ax4 = fig.add_subplot(gs[1, 1])
       for state in unique_states:
           state_returns = data.loc[states['hmm_state'] == state, 'returns']
           if len(state_returns) > 0:
               ax4.hist(state_returns, bins=20, alpha=0.6,
                       label=f'State {state}', density=True)

       ax4.set_title('Return Distribution by State', fontsize=12)
       ax4.set_xlabel('Returns')
       ax4.set_ylabel('Density')
       ax4.legend()
       ax4.grid(True, alpha=0.3)

       # Plot 5: Transition matrix
       ax5 = fig.add_subplot(gs[1, 2])
       n_states = len(unique_states)
       transition_matrix = np.zeros((n_states, n_states))

       for i in range(len(states) - 1):
           from_state = states['hmm_state'].iloc[i]
           to_state = states['hmm_state'].iloc[i + 1]
           transition_matrix[from_state, to_state] += 1

       # Normalize
       row_sums = transition_matrix.sum(axis=1, keepdims=True)
       transition_probs = np.divide(transition_matrix, row_sums,
                                    where=row_sums != 0, out=np.zeros_like(transition_matrix))

       im = ax5.imshow(transition_probs, cmap='Blues', aspect='auto')
       ax5.set_title('Transition Matrix', fontsize=12)
       ax5.set_xlabel('To State')
       ax5.set_ylabel('From State')
       ax5.set_xticks(range(n_states))
       ax5.set_yticks(range(n_states))

       # Add probability values
       for i in range(n_states):
           for j in range(n_states):
               if transition_probs[i, j] > 0.01:
                   text_color = 'white' if transition_probs[i, j] > 0.5 else 'black'
                   ax5.text(j, i, f'{transition_probs[i, j]:.2f}',
                           ha='center', va='center', color=text_color)

       # Plot 6: Risk-return scatter
       ax6 = fig.add_subplot(gs[2, :2])
       risk_return_data = []

       for state in unique_states:
           state_returns = data.loc[states['hmm_state'] == state, 'returns']
           if len(state_returns) > 0:
               risk_return_data.append({
                   'state': state,
                   'mean_return': state_returns.mean(),
                   'volatility': state_returns.std(),
                   'sharpe': state_returns.mean() / state_returns.std() * np.sqrt(252) if state_returns.std() > 0 else 0
               })

       risk_return_df = pd.DataFrame(risk_return_data)

       scatter = ax6.scatter(risk_return_df['volatility'], risk_return_df['mean_return'],
                            s=200, alpha=0.7, c=risk_return_df['state'], cmap='viridis')

       ax6.set_title('Risk-Return Profile by State', fontsize=12, fontweight='bold')
       ax6.set_xlabel('Volatility (Risk)')
       ax6.set_ylabel('Mean Return')

       for _, row in risk_return_df.iterrows():
           ax6.annotate(f"State {row['state']}\nSharpe: {row['sharpe']:.2f}",
                       (row['volatility'], row['mean_return']),
                       xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', alpha=0.7))

       ax6.grid(True, alpha=0.3)

       # Plot 7: State statistics table
       ax7 = fig.add_subplot(gs[2, 2])
       ax7.axis('tight')
       ax7.axis('off')

       # Create statistics table
       stats_data = []
       for state in unique_states:
           state_data = data[states['hmm_state'] == state]
           if len(state_data) > 0:
               stats_data.append([
                   f"State {state}",
                   f"{len(state_data)}",
                   f"{state_data['returns'].mean():.4f}",
                   f"{state_data['volatility_14'].mean():.4f}",
                   f"{len(state_data) / len(data) * 100:.1f}%"
               ])

       table = ax7.table(cellText=stats_data,
                        colLabels=['State', 'Periods', 'Mean Return', 'Mean Vol', '% Time'],
                        cellLoc='center',
                        loc='center')
       table.auto_set_font_size(False)
       table.set_fontsize(9)
       table.scale(1, 1.5)

       # Plot 8: Performance metrics
       ax8 = fig.add_subplot(gs[3, :])
       metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
       state_performance = []

       for state in unique_states:
           state_data = data[states['hmm_state'] == state]
           if len(state_data) > 0:
               returns = state_data['returns']
               cumulative_returns = (1 + returns).cumprod() - 1
               max_drawdown = (cumulative_returns - cumulative_returns.expanding().max()).min()

               state_performance.append([
                   f"State {state}",
                   f"{cumulative_returns.iloc[-1]:.3f}",
                   f"{returns.mean() / returns.std() * np.sqrt(252):.3f}" if returns.std() > 0 else "0.000",
                   f"{max_drawdown:.3f}",
                   f"{(returns > 0).mean():.2%}"
               ])

       performance_df = pd.DataFrame(state_performance,
                                    columns=['State', 'Total Return', 'Sharpe', 'Max DD', 'Win Rate'])

       # Create horizontal bar chart for key metrics
       y_pos = np.arange(len(performance_df))
       width = 0.15

       for i, metric in enumerate(['Total Return', 'Sharpe', 'Max DD']):
           values = pd.to_numeric(performance_df[metric], errors='coerce')
           bars = ax8.barh(y_pos + i * width, values, width,
                          label=metric, alpha=0.8)

           # Add value labels
           for j, bar in enumerate(bars):
               width_bar = bar.get_width()
               ax8.text(width_bar, bar.get_y() + bar.get_height()/2,
                       f'{width_bar:.3f}', ha='left', va='center', fontsize=9)

       ax8.set_yticks(y_pos + width)
       ax8.set_yticklabels(performance_df['State'])
       ax8.set_xlabel('Metric Value')
       ax8.set_title('Performance Metrics by State', fontsize=12, fontweight='bold')
       ax8.legend()
       ax8.grid(True, alpha=0.3, axis='x')

       plt.tight_layout()
       plt.show()

   # Create comprehensive dashboard
   create_hmm_dashboard(results, results, "Comprehensive HMM Analysis Dashboard")

Saving and Exporting
--------------------

Export Visualizations
~~~~~~~~~~~~~~~~~~~~~

Save charts in various formats for reports and presentations:

.. code-block:: python

   def save_visualizations(data, states, output_dir="./visualizations/"):
       """Save all key visualizations to files."""

       import os
       os.makedirs(output_dir, exist_ok=True)

       print(f"Saving visualizations to {output_dir}")

       # 1. Price chart with states
       fig, ax = plt.subplots(figsize=(15, 8))
       unique_states = sorted(states['hmm_state'].unique())
       colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
       state_colors = dict(zip(unique_states, colors))

       ax.plot(data['datetime'], data['close'], color='black', linewidth=2, label='Price')

       for i in range(len(states)):
           state = states['hmm_state'].iloc[i]
           if i == 0 or states['hmm_state'].iloc[i-1] != state:
               start_idx = i
           elif i == len(states) - 1 or states['hmm_state'].iloc[i+1] != state:
               end_idx = i
               ax.axvspan(data['datetime'].iloc[start_idx], data['datetime'].iloc[end_idx],
                          alpha=0.3, color=state_colors[state],
                          label=f'State {state}' if start_idx == 0 or states['hmm_state'].iloc[start_idx-1] != state else "")

       ax.set_title('Price with HMM States', fontsize=14, fontweight='bold')
       ax.set_ylabel('Price ($)')
       ax.legend()
       ax.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f"{output_dir}/price_with_states.png", dpi=300, bbox_inches='tight')
       plt.close()

       # 2. State distribution
       fig, ax = plt.subplots(figsize=(10, 6))
       state_counts = states['hmm_state'].value_counts().sort_index()
       colors_pie = [state_colors[state] for state in state_counts.index]
       ax.pie(state_counts.values, labels=[f'State {i}' for i in state_counts.index],
              autopct='%1.1f%%', colors=colors_pie, startangle=90)
       ax.set_title('State Distribution', fontsize=14, fontweight='bold')
       plt.tight_layout()
       plt.savefig(f"{output_dir}/state_distribution.png", dpi=300, bbox_inches='tight')
       plt.close()

       # 3. Return distributions
       fig, ax = plt.subplots(figsize=(12, 8))
       for state in unique_states:
           state_returns = data.loc[states['hmm_state'] == state, 'returns']
           if len(state_returns) > 0:
               ax.hist(state_returns, bins=30, alpha=0.6,
                      label=f'State {state}', density=True)

       ax.set_title('Return Distributions by State', fontsize=14, fontweight='bold')
       ax.set_xlabel('Returns')
       ax.set_ylabel('Density')
       ax.legend()
       ax.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f"{output_dir}/return_distributions.png", dpi=300, bbox_inches='tight')
       plt.close()

       # 4. Transition matrix
       fig, ax = plt.subplots(figsize=(8, 6))
       n_states = len(unique_states)
       transition_matrix = np.zeros((n_states, n_states))

       for i in range(len(states) - 1):
           from_state = states['hmm_state'].iloc[i]
           to_state = states['hmm_state'].iloc[i + 1]
           transition_matrix[from_state, to_state] += 1

       row_sums = transition_matrix.sum(axis=1, keepdims=True)
       transition_probs = np.divide(transition_matrix, row_sums,
                                    where=row_sums != 0, out=np.zeros_like(transition_matrix))

       im = ax.imshow(transition_probs, cmap='Blues', aspect='auto')
       ax.set_title('State Transition Matrix', fontsize=14, fontweight='bold')
       ax.set_xlabel('To State')
       ax.set_ylabel('From State')

       # Add colorbar
       cbar = plt.colorbar(im, ax=ax)
       cbar.set_label('Transition Probability')

       plt.tight_layout()
       plt.savefig(f"{output_dir}/transition_matrix.png", dpi=300, bbox_inches='tight')
       plt.close()

       # 5. Performance summary
       fig, ax = plt.subplots(figsize=(12, 8))
       performance_data = []

       for state in unique_states:
           state_data = data[states['hmm_state'] == state]
           if len(state_data) > 0:
               returns = state_data['returns']
               performance_data.append({
                   'state': state,
                   'total_return': (1 + returns).cumprod().iloc[-1] - 1,
                   'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                   'max_drawdown': ((1 + returns).cumprod() - (1 + returns).cumprod().expanding().max()).min(),
                   'win_rate': (returns > 0).mean()
               })

       perf_df = pd.DataFrame(performance_data)

       x = np.arange(len(perf_df))
       width = 0.2

       ax.bar(x - width*1.5, perf_df['total_return'], width, label='Total Return', alpha=0.8)
       ax.bar(x - width*0.5, perf_df['sharpe'], width, label='Sharpe Ratio', alpha=0.8)
       ax.bar(x + width*0.5, -perf_df['max_drawdown'], width, label='Max Drawdown (neg)', alpha=0.8)
       ax.bar(x + width*1.5, perf_df['win_rate'], width, label='Win Rate', alpha=0.8)

       ax.set_xlabel('State')
       ax.set_ylabel('Value')
       ax.set_title('Performance Metrics by State', fontsize=14, fontweight='bold')
       ax.set_xticks(x)
       ax.set_xticklabels([f"State {s}" for s in perf_df['state']])
       ax.legend()
       ax.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f"{output_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
       plt.close()

       print(f"✅ Saved 5 visualization files to {output_dir}")
       print("Files created:")
       print("  - price_with_states.png")
       print("  - state_distribution.png")
       print("  - return_distributions.png")
       print("  - transition_matrix.png")
       print("  - performance_metrics.png")

   # Save all visualizations
   save_visualizations(results, results)

This comprehensive visualization guide provides the tools to create professional, informative charts that effectively communicate HMM analysis results and support decision-making processes.