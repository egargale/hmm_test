"""
Report Generator Module

Implements detailed regime analysis report generation for HMM strategies,
producing professional PDF/HTML reports with comprehensive analysis.
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Jinja2 for templating
from jinja2 import Template

# Local imports
from backtesting.performance_metrics import calculate_returns
from utils import get_logger
from utils.data_types import BacktestResult, PerformanceMetrics

# WeasyPrint for PDF generation (optional)
WEASYPRINT_AVAILABLE = False

logger = get_logger(__name__)


def analyze_regime_characteristics(
    states: np.ndarray, indicators: pd.DataFrame, equity_curve: pd.Series
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze characteristics of each HMM regime.

    Args:
        states: HMM state sequence
        indicators: Technical indicators DataFrame
        equity_curve: Strategy equity curve

    Returns:
        Dictionary mapping states to their characteristics
    """
    logger.info("Analyzing regime characteristics")

    # Create combined DataFrame
    data_df = pd.DataFrame(
        {"state": states, "equity": equity_curve}, index=equity_curve.index
    )

    # Add indicators if available
    for col in indicators.columns:
        data_df[col] = indicators[col]

    # Calculate returns
    data_df["returns"] = calculate_returns(data_df["equity"])

    # Analyze each state
    regime_analysis = {}
    unique_states = np.unique(states)
    unique_states = unique_states[unique_states >= 0]  # Skip negative states

    for state in unique_states:
        state_data = data_df[data_df["state"] == state]

        if len(state_data) == 0:
            continue

        analysis = {
            "state_id": int(state),
            "sample_size": len(state_data),
            "percentage": len(state_data) / len(data_df) * 100,
            "duration_stats": calculate_duration_stats(states, state),
            "return_stats": calculate_return_stats(state_data["returns"]),
            "volatility_stats": calculate_volatility_stats(state_data["returns"]),
            "indicator_stats": {},
        }

        # Analyze indicators for this state
        for indicator in indicators.columns:
            if indicator in state_data.columns:
                indicator_data = state_data[indicator].dropna()
                if len(indicator_data) > 0:
                    analysis["indicator_stats"][indicator] = {
                        "mean": float(indicator_data.mean()),
                        "std": float(indicator_data.std()),
                        "min": float(indicator_data.min()),
                        "max": float(indicator_data.max()),
                        "median": float(indicator_data.median()),
                    }

        regime_analysis[state] = analysis

    logger.info(f"Analyzed {len(regime_analysis)} regimes")
    return regime_analysis


def calculate_duration_stats(states: np.ndarray, target_state: int) -> Dict[str, float]:
    """
    Calculate duration statistics for a specific state.

    Args:
        states: HMM state sequence
        target_state: State to analyze

    Returns:
        Dictionary with duration statistics
    """
    # Find consecutive periods of the target state
    durations = []
    current_duration = 0

    for state in states:
        if state == target_state:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

    # Add final duration if sequence ends with target state
    if current_duration > 0:
        durations.append(current_duration)

    if not durations:
        return {
            "mean_duration": 0.0,
            "median_duration": 0.0,
            "max_duration": 0.0,
            "min_duration": 0.0,
            "std_duration": 0.0,
            "total_periods": 0,
        }

    return {
        "mean_duration": float(np.mean(durations)),
        "median_duration": float(np.median(durations)),
        "max_duration": float(np.max(durations)),
        "min_duration": float(np.min(durations)),
        "std_duration": float(np.std(durations)),
        "total_periods": len(durations),
    }


def calculate_return_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate return statistics for a state.

    Args:
        returns: Returns series for the state

    Returns:
        Dictionary with return statistics
    """
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        return {
            "mean_daily_return": 0.0,
            "annualized_return": 0.0,
            "return_std": 0.0,
            "return_skewness": 0.0,
            "return_kurtosis": 0.0,
            "positive_return_pct": 0.0,
            "negative_return_pct": 0.0,
        }

    return {
        "mean_daily_return": float(returns_clean.mean()),
        "annualized_return": float(returns_clean.mean() * 252),
        "return_std": float(returns_clean.std()),
        "return_skewness": float(returns_clean.skew()),
        "return_kurtosis": float(returns_clean.kurtosis()),
        "positive_return_pct": float((returns_clean > 0).mean() * 100),
        "negative_return_pct": float((returns_clean < 0).mean() * 100),
    }


def calculate_volatility_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate volatility statistics for a state.

    Args:
        returns: Returns series for the state

    Returns:
        Dictionary with volatility statistics
    """
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        return {
            "daily_volatility": 0.0,
            "annualized_volatility": 0.0,
            "downside_volatility": 0.0,
            "volatility_of_volatility": 0.0,
        }

    daily_vol = returns_clean.std()
    annualized_vol = daily_vol * np.sqrt(252)

    # Downside volatility (returns < 0)
    downside_returns = returns_clean[returns_clean < 0]
    downside_vol = (
        downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    )

    # Volatility of volatility (rolling std)
    rolling_vol = returns_clean.rolling(window=21).std().dropna()
    vol_of_vol = rolling_vol.std() if len(rolling_vol) > 0 else 0.0

    return {
        "daily_volatility": float(daily_vol),
        "annualized_volatility": float(annualized_vol),
        "downside_volatility": float(downside_vol),
        "volatility_of_volatility": float(vol_of_vol),
    }


def calculate_transition_matrix(states: np.ndarray) -> pd.DataFrame:
    """
    Calculate transition probability matrix for HMM states.

    Args:
        states: HMM state sequence

    Returns:
        Transition probability matrix as DataFrame
    """
    logger.info("Calculating transition matrix")

    unique_states = np.unique(states)
    unique_states = unique_states[unique_states >= 0]  # Skip negative states

    n_states = len(unique_states)
    transition_matrix = np.zeros((n_states, n_states))

    # Count transitions
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]

        if current_state >= 0 and next_state >= 0:  # Skip negative states
            current_idx = np.where(unique_states == current_state)[0][0]
            next_idx = np.where(unique_states == next_state)[0][0]
            transition_matrix[current_idx, next_idx] += 1

    # Convert to probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]
    transition_matrix = np.nan_to_num(transition_matrix)

    # Create DataFrame
    transition_df = pd.DataFrame(
        transition_matrix,
        index=[f"State {int(s)}" for s in unique_states],
        columns=[f"State {int(s)}" for s in unique_states],
    )

    return transition_df


def create_regime_charts_data(
    regime_analysis: Dict[int, Dict[str, Any]], transition_matrix: pd.DataFrame
) -> Dict[str, str]:
    """
    Create base64 encoded charts for the report.

    Args:
        regime_analysis: Regime characteristics analysis
        transition_matrix: Transition probability matrix

    Returns:
        Dictionary with base64 encoded chart data
    """
    logger.info("Creating charts for report")

    charts_data = {}

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. State Distribution Pie Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        state_labels = [f"State {k}" for k in regime_analysis.keys()]
        state_sizes = [v["percentage"] for v in regime_analysis.values()]

        ax.pie(state_sizes, labels=state_labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Regime Distribution", fontsize=14, fontweight="bold")

        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        charts_data["state_distribution"] = base64.b64encode(img_buffer.read()).decode()
        plt.close()

        # 2. Transition Matrix Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            transition_matrix,
            annot=True,
            cmap="Blues",
            square=True,
            fmt=".2f",
            cbar_kws={"label": "Transition Probability"},
        )
        ax.set_title("State Transition Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")

        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        charts_data["transition_matrix"] = base64.b64encode(img_buffer.read()).decode()
        plt.close()

        # 3. Regime Performance Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        states = list(regime_analysis.keys())
        returns = [
            regime_analysis[s]["return_stats"]["annualized_return"] for s in states
        ]
        volatilities = [
            regime_analysis[s]["volatility_stats"]["annualized_volatility"]
            for s in states
        ]
        durations = [
            regime_analysis[s]["duration_stats"]["mean_duration"] for s in states
        ]
        positive_returns = [
            regime_analysis[s]["return_stats"]["positive_return_pct"] for s in states
        ]

        # Annualized Returns
        ax1.bar(range(len(states)), returns, color="skyblue")
        ax1.set_title("Annualized Returns by Regime")
        ax1.set_ylabel("Annualized Return")
        ax1.set_xticks(range(len(states)))
        ax1.set_xticklabels([f"State {s}" for s in states])
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Volatility
        ax2.bar(range(len(states)), volatilities, color="lightcoral")
        ax2.set_title("Annualized Volatility by Regime")
        ax2.set_ylabel("Annualized Volatility")
        ax2.set_xticks(range(len(states)))
        ax2.set_xticklabels([f"State {s}" for s in states])

        # Average Duration
        ax3.bar(range(len(states)), durations, color="lightgreen")
        ax3.set_title("Average Duration by Regime")
        ax3.set_ylabel("Average Duration (periods)")
        ax3.set_xticks(range(len(states)))
        ax3.set_xticklabels([f"State {s}" for s in states])

        # Positive Return Percentage
        ax4.bar(range(len(states)), positive_returns, color="gold")
        ax4.set_title("Positive Return Percentage by Regime")
        ax4.set_ylabel("Positive Returns (%)")
        ax4.set_xticks(range(len(states)))
        ax4.set_xticklabels([f"State {s}" for s in states])

        plt.tight_layout()

        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        charts_data["regime_performance"] = base64.b64encode(img_buffer.read()).decode()
        plt.close()

        logger.info("Created all charts successfully")

    except ImportError as e:
        logger.warning(f"Could not create charts due to missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Failed to create charts: {e}")

    return charts_data


def generate_html_report(
    regime_analysis: Dict[int, Dict[str, Any]],
    transition_matrix: pd.DataFrame,
    metrics: PerformanceMetrics,
    result: BacktestResult,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate HTML report content.

    Args:
        regime_analysis: Regime characteristics analysis
        transition_matrix: Transition probability matrix
        metrics: Performance metrics
        result: Backtest results
        config: Report configuration

    Returns:
        HTML content as string
    """
    # Create charts data
    charts_data = create_regime_charts_data(regime_analysis, transition_matrix)

    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid #e9ecef;
            }
            .header h1 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .header .subtitle {
                color: #6c757d;
                font-size: 1.1em;
            }
            .section {
                margin-bottom: 40px;
            }
            .section h2 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .section h3 {
                color: #34495e;
                margin-bottom: 15px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }
            .metric-card .metric-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #2c3e50;
            }
            .metric-card .metric-label {
                color: #6c757d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .chart-container {
                text-align: center;
                margin: 30px 0;
            }
            .chart-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .regime-details {
                margin-bottom: 30px;
            }
            .regime-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .regime-table th,
            .regime-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            .regime-table th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }
            .regime-table tr:hover {
                background-color: #f8f9fa;
            }
            .transition-matrix {
                margin: 20px 0;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #e9ecef;
                color: #6c757d;
            }
            .positive { color: #27ae60; font-weight: bold; }
            .negative { color: #e74c3c; font-weight: bold; }
            .highlight { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{{ title }}</h1>
                <div class="subtitle">
                    Comprehensive Analysis of HMM Trading Strategy Performance<br>
                    Generated on {{ timestamp }}
                </div>
            </div>

            <!-- Executive Summary -->
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="highlight">
                    <strong>Strategy Overview:</strong> The HMM-based trading strategy has been analyzed across {{ num_regimes }} distinct market regimes.
                    The strategy achieved a total return of <span class="{% if metrics.total_return > 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(metrics.total_return * 100) }}%</span>
                    with a Sharpe ratio of {{ "%.2f"|format(metrics.sharpe_ratio) }}.
                </div>

                <h3>Key Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.total_return * 100) }}%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.annualized_return * 100) }}%</div>
                        <div class="metric-label">Annualized Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.sharpe_ratio) }}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.max_drawdown * 100) }}%</div>
                        <div class="metric-label">Maximum Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.win_rate * 100) }}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.profit_factor) }}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                </div>
            </div>

            <!-- Regime Analysis -->
            <div class="section">
                <h2>Regime Analysis</h2>

                {% if charts_data['state_distribution'] %}
                <div class="chart-container">
                    <h3>Regime Distribution</h3>
                    <img src="data:image/png;base64,{{ charts_data['state_distribution'] }}" alt="State Distribution">
                </div>
                {% endif %}

                {% if charts_data['transition_matrix'] %}
                <div class="chart-container">
                    <h3>State Transition Matrix</h3>
                    <img src="data:image/png;base64,{{ charts_data['transition_matrix'] }}" alt="Transition Matrix">
                    <div class="transition-matrix">
                        <table class="regime-table">
                            <thead>
                                <tr>
                                    <th>From / To</th>
                                    {% for state in transition_matrix.columns %}
                                    <th>{{ state }}</th>
                                    {% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for state in transition_matrix.index %}
                                <tr>
                                    <th>{{ state }}</th>
                                    {% for prob in transition_matrix.loc[state] %}
                                    <td>{{ "%.2f"|format(prob * 100) }}%</td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                {% endif %}

                {% if charts_data['regime_performance'] %}
                <div class="chart-container">
                    <h3>Regime Performance Comparison</h3>
                    <img src="data:image/png;base64,{{ charts_data['regime_performance'] }}" alt="Regime Performance">
                </div>
                {% endif %}
            </div>

            <!-- Detailed Regime Characteristics -->
            <div class="section">
                <h2>Detailed Regime Characteristics</h2>

                {% for state_id, analysis in regime_analysis.items() %}
                <div class="regime-details">
                    <h3>State {{ state_id }} Characteristics</h3>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.1f"|format(analysis.percentage) }}%</div>
                            <div class="metric-label">Time in Regime</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.1f"|format(analysis.duration_stats.mean_duration) }}</div>
                            <div class="metric-label">Avg Duration (periods)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {{ 'positive' if analysis.return_stats.annualized_return > 0 else 'negative' }}">
                                {{ "%.2f"|format(analysis.return_stats.annualized_return * 100) }}%
                            </div>
                            <div class="metric-label">Annualized Return</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format(analysis.volatility_stats.annualized_volatility * 100) }}%</div>
                            <div class="metric-label">Annualized Volatility</div>
                        </div>
                    </div>

                    <table class="regime-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Sample Size</td>
                                <td>{{ analysis.sample_size }} periods</td>
                            </tr>
                            <tr>
                                <td>Mean Daily Return</td>
                                <td class="{{ 'positive' if analysis.return_stats.mean_daily_return > 0 else 'negative' }}">
                                    {{ "%.4f"|format(analysis.return_stats.mean_daily_return * 100) }}%
                                </td>
                            </tr>
                            <tr>
                                <td>Return Standard Deviation</td>
                                <td>{{ "%.4f"|format(analysis.return_stats.return_std * 100) }}%</td>
                            </tr>
                            <tr>
                                <td>Positive Return Percentage</td>
                                <td>{{ "%.1f"|format(analysis.return_stats.positive_return_pct) }}%</td>
                            </tr>
                            <tr>
                                <td>Downside Volatility</td>
                                <td>{{ "%.2f"|format(analysis.volatility_stats.downside_volatility * 100) }}%</td>
                            </tr>
                            <tr>
                                <td>Max Duration</td>
                                <td>{{ analysis.duration_stats.max_duration }} periods</td>
                            </tr>
                            <tr>
                                <td>Median Duration</td>
                                <td>{{ "%.1f"|format(analysis.duration_stats.median_duration) }} periods</td>
                            </tr>
                        </tbody>
                    </table>

                    {% if analysis.indicator_stats %}
                    <h4>Technical Indicator Statistics</h4>
                    <table class="regime-table">
                        <thead>
                            <tr>
                                <th>Indicator</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>Max</th>
                                <th>Median</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for indicator, stats in analysis.indicator_stats.items() %}
                            <tr>
                                <td>{{ indicator }}</td>
                                <td>{{ "%.4f"|format(stats.mean) }}</td>
                                <td>{{ "%.4f"|format(stats.std) }}</td>
                                <td>{{ "%.4f"|format(stats.min) }}</td>
                                <td>{{ "%.4f"|format(stats.max) }}</td>
                                <td>{{ "%.4f"|format(stats.median) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% endif %}
                </div>
                {% endfor %}
            </div>

            <!-- Conclusions and Recommendations -->
            <div class="section">
                <h2>Conclusions and Recommendations</h2>

                <div class="highlight">
                    <h3>Key Findings:</h3>
                    <ul>
                        {% for state_id, analysis in regime_analysis.items() %}
                        <li><strong>State {{ state_id }}:</strong>
                            {% if analysis.return_stats.annualized_return > 0 %}
                            Bullish regime with {{ "%.2f"|format(analysis.return_stats.annualized_return * 100) }}% annualized return
                            {% else %}
                            Bearish regime with {{ "%.2f"|format(analysis.return_stats.annualized_return * 100) }}% annualized return
                            {% endif %}
                            and {{ "%.2f"|format(analysis.volatility_stats.annualized_volatility * 100) }}% volatility.
                            Present in {{ "%.1f"|format(analysis.percentage) }}% of the observed period.
                        </li>
                        {% endfor %}
                    </ul>
                </div>

                <div class="highlight">
                    <h3>Risk Management Insights:</h3>
                    <ul>
                        <li>Maximum drawdown of {{ "%.2f"|format(metrics.max_drawdown * 100) }}% suggests
                           {% if metrics.max_drawdown < 0.1 %}excellent{% elif metrics.max_drawdown < 0.2 %}good{% else %}concerning{% endif %}
                           risk control.</li>
                        <li>Sharpe ratio of {{ "%.2f"|format(metrics.sharpe_ratio) }} indicates
                           {% if metrics.sharpe_ratio > 1.5 %}excellent{% elif metrics.sharpe_ratio > 1.0 %}good{% elif metrics.sharpe_ratio > 0.5 %}moderate{% else %}poor{% endif %}
                           risk-adjusted performance.</li>
                        <li>Win rate of {{ "%.1f"|format(metrics.win_rate * 100) }}% shows
                           {% if metrics.win_rate > 0.6 %}high{% elif metrics.win_rate > 0.5 %}moderate{% else %}low{% endif %}
                           trade success rate.</li>
                    </ul>
                </div>
            </div>

            <div class="footer">
                <p><em>Report generated by HMM Strategy Analysis System on {{ timestamp }}</em></p>
                <p>This report provides comprehensive analysis of Hidden Markov Model-based trading strategy performance across identified market regimes.</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Prepare template data
    template_data = {
        "title": config.get("title", "HMM Strategy Regime Analysis Report")
        if config
        else "HMM Strategy Regime Analysis Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_regimes": len(regime_analysis),
        "metrics": metrics,
        "regime_analysis": regime_analysis,
        "transition_matrix": transition_matrix,
        "charts_data": charts_data,
    }

    # Render template
    template = Template(html_template)
    return template.render(**template_data)


def generate_regime_report(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    states: np.ndarray,
    indicators: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    format: str = "html",
) -> str:
    """
    Generate a detailed regime analysis report.

    Args:
        result: Backtest results
        metrics: Performance metrics
        states: HMM state sequence
        indicators: Technical indicators DataFrame
        config: Report configuration
        output_path: Optional path to save the report
        format: Output format ('html' or 'pdf')

    Returns:
        Path to the saved report file
    """
    logger.info(f"Generating {format.upper()} regime analysis report")

    # Default configuration
    default_config = {
        "title": "HMM Strategy Regime Analysis Report",
        "include_charts": True,
        "include_indicators": True,
        "chart_style": "seaborn",
    }

    if config:
        default_config.update(config)
    config = default_config

    try:
        # Analyze regime characteristics
        indicators_df = indicators if indicators is not None else pd.DataFrame()
        regime_analysis = analyze_regime_characteristics(
            states, indicators_df, result.equity_curve
        )

        # Calculate transition matrix
        transition_matrix = calculate_transition_matrix(states)

        # Generate HTML report
        html_content = generate_html_report(
            regime_analysis, transition_matrix, metrics, result, config
        )

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"hmm_regime_analysis_{timestamp}.{format}"

        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save report
        if format.lower() == "pdf":
            # Try to import WeasyPrint only when needed
            try:
                from weasyprint import CSS, HTML

                # Convert HTML to PDF
                html_doc = HTML(string=html_content)
                css = CSS(
                    string="""
                    @page {
                        size: A4;
                        margin: 2cm;
                    }
                    body {
                        font-size: 10pt;
                    }
                """
                )
                html_doc.write_pdf(output_path, stylesheets=[css])
            except ImportError as e:
                raise ImportError(
                    "WeasyPrint is not available. Please install system dependencies or use HTML format."
                ) from e
        else:
            # Save as HTML
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        logger.info(f"Regime analysis report saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to generate regime report: {e}")
        raise
