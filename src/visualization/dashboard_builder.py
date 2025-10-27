"""
Dashboard Builder Module

Implements interactive HTML dashboard generation for HMM strategy performance
analysis using Plotly for interactive visualizations.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting.performance_analyzer import calculate_drawdown
from backtesting.performance_metrics import calculate_returns
from utils import get_logger
from utils.data_types import BacktestResult, PerformanceMetrics

logger = get_logger(__name__)


def create_equity_curve_chart(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create an interactive equity curve chart.

    Args:
        equity_curve: Strategy equity curve series
        benchmark_curve: Optional benchmark equity curve
        config: Configuration dictionary

    Returns:
        Plotly figure object
    """
    default_config = {
        'title': 'Strategy Equity Curve',
        'yaxis_title': 'Portfolio Value',
        'xaxis_title': 'Date',
        'height': 500,
        'show_legend': True,
        'line_width': 2
    }

    if config:
        default_config.update(config)
    config = default_config

    fig = go.Figure()

    # Add strategy equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line={'color': 'blue', 'width': config['line_width']},
        hovertemplate='<b>Strategy</b><br>' +
                     'Date: %{x|%Y-%m-%d}<br>' +
                     'Value: $%{y:,.2f}<extra></extra>'
    ))

    # Add benchmark if provided
    if benchmark_curve is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve.values,
            mode='lines',
            name='Benchmark',
            line={'color': 'gray', 'width': config['line_width'], 'dash': 'dash'},
            hovertemplate='<b>Benchmark</b><br>' +
                         'Date: %{x|%Y-%m-%d}<br>' +
                         'Value: $%{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=config['title'],
        xaxis_title=config['xaxis_title'],
        yaxis_title=config['yaxis_title'],
        height=config['height'],
        showlegend=config['show_legend'],
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_drawdown_chart(
    equity_curve: pd.Series,
    config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create an interactive drawdown chart.

    Args:
        equity_curve: Strategy equity curve series
        config: Configuration dictionary

    Returns:
        Plotly figure object
    """
    default_config = {
        'title': 'Strategy Drawdown',
        'yaxis_title': 'Drawdown (%)',
        'xaxis_title': 'Date',
        'height': 400,
        'fill_color': 'red',
        'fill_alpha': 0.3,
        'line_color': 'darkred'
    }

    if config:
        default_config.update(config)
    config = default_config

    # Calculate drawdown
    drawdown = calculate_drawdown(equity_curve) * 100

    fig = go.Figure()

    # Add drawdown area
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        fill='tonexty',
        fillcolor=f'rgba(220, 53, 69, {config["fill_alpha"]})',
        line={'color': config['line_color'], 'width': 1.5},
        hovertemplate='<b>Drawdown</b><br>' +
                     'Date: %{x|%Y-%m-%d}<br>' +
                     'Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=config['title'],
        xaxis_title=config['xaxis_title'],
        yaxis_title=config['yaxis_title'],
        height=config['height'],
        showlegend=False,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_returns_distribution_chart(
    returns: pd.Series,
    config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create an interactive returns distribution chart.

    Args:
        returns: Returns series
        config: Configuration dictionary

    Returns:
        Plotly figure object
    """
    default_config = {
        'title': 'Returns Distribution',
        'xaxis_title': 'Daily Return (%)',
        'yaxis_title': 'Frequency',
        'height': 400,
        'bins': 50,
        'color': 'blue'
    }

    if config:
        default_config.update(config)
    config = default_config

    returns_pct = returns * 100

    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=returns_pct,
        nbinsx=config['bins'],
        name='Returns',
        marker_color=config['color'],
        opacity=0.7,
        hovertemplate='<b>Returns</b><br>' +
                     'Range: %{x:.2f}% to %{y:.2f}%<br>' +
                     'Count: %{z}<extra></extra>'
    ))

    # Add vertical lines for statistics
    mean_return = returns_pct.mean()
    std_return = returns_pct.std()

    fig.add_vline(x=mean_return, line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {mean_return:.2f}%")
    fig.add_vline(x=mean_return + std_return, line_dash="dash", line_color="orange",
                  annotation_text=f"+1σ: {(mean_return + std_return):.2f}%")
    fig.add_vline(x=mean_return - std_return, line_dash="dash", line_color="orange",
                  annotation_text=f"-1σ: {(mean_return - std_return):.2f}%")

    fig.update_layout(
        title=config['title'],
        xaxis_title=config['xaxis_title'],
        yaxis_title=config['yaxis_title'],
        height=config['height'],
        showlegend=False,
        template='plotly_white'
    )

    return fig


def create_monthly_returns_heatmap(
    equity_curve: pd.Series,
    config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a monthly returns heatmap.

    Args:
        equity_curve: Strategy equity curve series
        config: Configuration dictionary

    Returns:
        Plotly figure object
    """
    default_config = {
        'title': 'Monthly Returns Heatmap',
        'height': 400,
        'colorscale': 'RdYlGn',
        'show_scale': True
    }

    if config:
        default_config.update(config)
    config = default_config

    # Calculate monthly returns
    returns = calculate_returns(equity_curve)
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table for heatmap
    monthly_returns_df = monthly_returns.to_frame('returns')
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month

    pivot_table = monthly_returns_df.pivot_table(
        values='returns',
        index='month',
        columns='year',
        fill_value=0
    )

    # Convert to percentage
    pivot_table = pivot_table * 100

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=[month_names[i-1] for i in pivot_table.index],
        colorscale=config['colorscale'],
        hoverongaps=False,
        hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=config['title'],
        xaxis_title='Year',
        yaxis_title='Month',
        height=config['height'],
        template='plotly_white'
    )

    if config['show_scale']:
        fig.update_layout(coloraxis_colorbar={'title': "Return (%)"})

    return fig


def create_regime_analysis_chart(
    states: np.ndarray,
    equity_curve: pd.Series,
    config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a regime analysis chart showing performance by HMM state.

    Args:
        states: HMM state sequence
        equity_curve: Strategy equity curve series
        config: Configuration dictionary

    Returns:
        Plotly figure object
    """
    default_config = {
        'title': 'Regime Performance Analysis',
        'height': 600,
        'show_bands': True,
        'band_alpha': 0.2
    }

    if config:
        default_config.update(config)
    config = default_config

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Equity Curve by Regime', 'Regime Timeline', 'Returns Distribution by Regime'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.2, 0.3]
    )

    unique_states = np.unique(states)
    unique_states = unique_states[unique_states >= 0]  # Skip negative states

    colors = px.colors.qualitative.Set1[:len(unique_states)]

    # Plot 1: Equity curve with regime backgrounds
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity Curve',
            line={'color': 'black', 'width': 2},
            hovertemplate='<b>Equity</b><br>Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add regime backgrounds
    if config['show_bands']:
        for i, state_value in enumerate(unique_states):
            state_mask = states == state_value
            color = colors[i % len(colors)]

            # Create background bands
            fig.add_vrect(
                x0=equity_curve.index[state_mask][0] if state_mask.any() else equity_curve.index[0],
                x1=equity_curve.index[state_mask][-1] if state_mask.any() else equity_curve.index[-1],
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {config['band_alpha']})",
                layer="below", line_width=0,
                row=1, col=1
            )

    # Plot 2: Regime timeline
    for i, state_value in enumerate(unique_states):
        state_mask = states == state_value
        color = colors[i % len(colors)]

        # Create regime timeline
        regime_values = np.where(state_mask, state_value, np.nan)

        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=regime_values,
                mode='markers',
                name=f'State {state_value}',
                marker={'color': color, 'size': 4},
                hovertemplate=f'<b>State {state_value}</b><br>Date: %{{x|%Y-%m-%d}}<extra></extra>'
            ),
            row=2, col=1
        )

    # Plot 3: Returns distribution by regime
    returns = calculate_returns(equity_curve)
    returns_df = pd.DataFrame({'returns': returns, 'state': states}, index=equity_curve.index)

    for i, state_value in enumerate(unique_states):
        state_returns = returns_df[returns_df['state'] == state_value]['returns'].dropna()
        if len(state_returns) > 0:
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Histogram(
                    x=state_returns * 100,
                    name=f'State {state_value}',
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=30,
                    hovertemplate=f'<b>State {state_value}</b><br>Return: %{{x:.2f}}%<extra></extra>'
                ),
                row=3, col=1
            )

    fig.update_layout(
        title=config['title'],
        height=config['height'],
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Update subplot titles
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="HMM State", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Daily Return (%)", row=3, col=1)

    return fig


def create_performance_metrics_table(
    metrics: PerformanceMetrics,
    config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a performance metrics table.

    Args:
        metrics: PerformanceMetrics object
        config: Configuration dictionary

    Returns:
        Plotly figure object
    """
    default_config = {
        'title': 'Performance Metrics Summary',
        'height': 400
    }

    if config:
        default_config.update(config)
    config = default_config

    # Prepare metrics data
    metrics_data = [
        ['Total Return', f"{metrics.total_return:.2%}"],
        ['Annualized Return', f"{metrics.annualized_return:.2%}"],
        ['Annualized Volatility', f"{metrics.annualized_volatility:.2%}"],
        ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"],
        ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}"],
        ['Maximum Drawdown', f"{metrics.max_drawdown:.2%}"],
        ['Max Drawdown Duration', f"{metrics.max_drawdown_duration} days"],
        ['Calmar Ratio', f"{metrics.calmar_ratio:.2f}"],
        ['Win Rate', f"{metrics.win_rate:.2%}"],
        ['Profit Factor', f"{metrics.profit_factor:.2f}"]
    ]

    # Create table
    fig = go.Figure(data=[go.Table(
        header={
            'values': ['Metric', 'Value'],
            'fill_color': 'lightblue',
            'align': 'left',
            'font': {'size': 14, 'color': 'black'}
        },
        cells={
            'values': [[row[0] for row in metrics_data],
                    [row[1] for row in metrics_data]],
            'fill_color': 'white',
            'align': 'left',
            'font': {'size': 12, 'color': 'black'}
        }
    )])

    fig.update_layout(
        title=config['title'],
        height=config['height'],
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20}
    )

    return fig


def build_dashboard(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    states: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Build an interactive HTML dashboard for HMM strategy performance analysis.

    Args:
        result: Backtest results
        metrics: Performance metrics
        states: HMM state sequence
        config: Dashboard configuration
        output_path: Optional path to save the HTML file

    Returns:
        Path to the saved HTML dashboard
    """
    logger.info("Building interactive performance dashboard")

    # Default configuration
    default_config = {
        'title': 'HMM Strategy Performance Dashboard',
        'include_regime_analysis': True,
        'include_monthly_heatmap': True,
        'include_distribution': True,
        'theme': 'plotly_white',
        'auto_open': False,
        'config': {'displayModeBar': True}
    }

    if config:
        default_config.update(config)
    config = default_config

    try:
        # Create individual chart components
        charts = {}

        # 1. Equity Curve
        charts['equity_curve'] = create_equity_curve_chart(
            result.equity_curve,
            config=config.get('equity_curve_config')
        )

        # 2. Drawdown Chart
        charts['drawdown'] = create_drawdown_chart(
            result.equity_curve,
            config=config.get('drawdown_config')
        )

        # 3. Returns Distribution
        if config['include_distribution']:
            returns = calculate_returns(result.equity_curve)
            charts['returns_distribution'] = create_returns_distribution_chart(
                returns,
                config=config.get('distribution_config')
            )

        # 4. Monthly Returns Heatmap
        if config['include_monthly_heatmap']:
            charts['monthly_heatmap'] = create_monthly_returns_heatmap(
                result.equity_curve,
                config=config.get('heatmap_config')
            )

        # 5. Regime Analysis
        if config['include_regime_analysis']:
            charts['regime_analysis'] = create_regime_analysis_chart(
                states,
                result.equity_curve,
                config=config.get('regime_config')
            )

        # 6. Performance Metrics Table
        charts['metrics_table'] = create_performance_metrics_table(
            metrics,
            config=config.get('metrics_config')
        )

        # Build HTML dashboard
        html_content = build_html_dashboard(charts, config)

        # Generate output path if not provided
        if output_path is None:
            output_path = "hmm_performance_dashboard.html"

        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save dashboard
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Interactive dashboard saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to build dashboard: {e}")
        raise


def build_html_dashboard(charts: Dict[str, go.Figure], config: Dict[str, Any]) -> str:
    """
    Build HTML dashboard from Plotly charts.

    Args:
        charts: Dictionary of chart names to Plotly figures
        config: Dashboard configuration

    Returns:
        HTML content as string
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                margin-bottom: 30px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 20px;
            }}
            .chart-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
            }}
            .full-width {{
                grid-column: 1 / -1;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on {timestamp}</p>
        </div>

        <div class="grid">
            {charts_html}
        </div>

        <div class="footer">
            <p>HMM Strategy Performance Dashboard | Generated by Automated Analysis System</p>
        </div>
    </body>
    </html>
    """

    # Generate HTML for each chart
    charts_html = ""
    chart_order = [
        ('equity_curve', 'Strategy Performance', 'full-width'),
        ('drawdown', 'Drawdown Analysis', 'full-width'),
        ('metrics_table', 'Performance Summary', ''),
        ('regime_analysis', 'Regime Analysis', 'full-width'),
        ('monthly_heatmap', 'Monthly Returns Heatmap', ''),
        ('returns_distribution', 'Returns Distribution', '')
    ]

    for chart_id, chart_title, css_class in chart_order:
        if chart_id in charts:
            chart_html = f"""
            <div class="chart-container {css_class}">
                <div class="chart-title">{chart_title}</div>
                <div id="{chart_id}"></div>
            </div>
            """
            charts_html += chart_html

    # Get current timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Fill template
    html_content = html_template.format(
        title=config['title'],
        timestamp=timestamp,
        charts_html=charts_html
    )

    # Add Plotly.js charts
    chart_scripts = "<script>\n"
    for chart_id, fig in charts.items():
        # Convert figure to JSON
        fig_json = fig.to_json()
        chart_scripts += f"""
        Plotly.newPlot('{chart_id}', {fig_json});
        """
    chart_scripts += "\n</script>"

    # Insert scripts before closing body tag
    html_content = html_content.replace("</body>", f"{chart_scripts}</body>")

    return html_content
