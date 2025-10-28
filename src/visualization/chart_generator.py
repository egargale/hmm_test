"""
Chart Generator Module

Implements state visualization engine for generating publication-ready charts
with HMM states overlaid on price data and technical indicators.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


def create_state_color_map(n_states: int, colormap: str = "tab10") -> Dict[int, str]:
    """
    Create a color map for HMM states.

    Args:
        n_states: Number of HMM states
        colormap: Matplotlib colormap name

    Returns:
        Dictionary mapping state indices to colors
    """
    if n_states <= 0:
        return {}

    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / max(n_states - 1, 1)) for i in range(n_states)]

    # Convert colors to hex strings for mplfinance
    hex_colors = []
    for color in colors:
        if isinstance(color, tuple) and len(color) >= 3:
            rgb = color[:3]  # Take RGB, ignore alpha if present
            hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
            hex_colors.append(hex_color)
        else:
            hex_colors.append(color)

    return {i: hex_colors[i] for i in range(n_states)}


def create_state_overlay_series(
    price_data: pd.Series,
    states: np.ndarray,
    state_color: str,
    state_value: int,
    alpha: float = 0.3,
) -> pd.Series:
    """
    Create a series for overlaying a specific HMM state on price data.

    Args:
        price_data: Price series (typically close prices)
        states: HMM state sequence
        state_color: Color for this state
        state_value: State value to overlay
        alpha: Transparency level

    Returns:
        Series with values only where the specified state is active
    """
    state_mask = states == state_value
    overlay_series = price_data.copy()

    # Set values to NaN where state is not active
    overlay_series[~state_mask] = np.nan

    return overlay_series


def create_state_background_plots(
    price_data: pd.DataFrame,
    states: np.ndarray,
    color_map: Dict[int, str],
    alpha: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Create background color plots for HMM states.

    Args:
        price_data: OHLCV price data
        states: HMM state sequence
        color_map: State color mapping
        alpha: Transparency level

    Returns:
        List of plot dictionaries for mplfinance
    """
    background_plots = []

    for state_value, color in color_map.items():
        state_mask = states == state_value

        # Create background bands
        background = pd.Series(index=price_data.index, dtype=float)

        # Fill with high values where state is active
        background[state_mask] = price_data["high"].max() * 1.1
        background[~state_mask] = np.nan

        # Create fill_between plot
        background_plots.append(
            {
                "type": "fill_between",
                "y": background,
                "color": color,
                "alpha": alpha,
                "panel": 0,
            }
        )

    return background_plots


def create_indicator_plots(
    indicators: pd.DataFrame,
    indicator_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Create plots for technical indicators.

    Args:
        indicators: DataFrame with technical indicators
        indicator_config: Configuration for each indicator

    Returns:
        List of plot dictionaries for mplfinance
    """
    if indicator_config is None:
        indicator_config = {
            "RSI": {"panel": 1, "color": "purple", "title": "RSI"},
            "MACD": {"panel": 2, "color": "blue", "title": "MACD"},
            "Volume": {"panel": 3, "color": "green", "title": "Volume"},
        }

    indicator_plots = []

    for indicator_name, config in indicator_config.items():
        if indicator_name in indicators.columns:
            indicator_plots.append(
                {
                    "type": "line",
                    "y": indicators[indicator_name],
                    "color": config.get("color", "black"),
                    "panel": config.get("panel", 1),
                    "title": config.get("title", indicator_name),
                }
            )

    return indicator_plots


def plot_states(
    price_data: pd.DataFrame,
    states: np.ndarray,
    indicators: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    Generate publication-ready charts with HMM states overlaid on price data.

    Args:
        price_data: OHLCV price data with datetime index
        states: HMM state sequence (same length as price_data)
        indicators: Optional DataFrame with technical indicators
        config: Configuration dictionary for plotting
        output_path: Optional path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Path to the saved plot file

    Raises:
        ValueError: If input data validation fails
    """
    logger.info("Starting HMM state visualization")

    # Default configuration
    default_config = {
        "chart_type": "candle",
        "style": "yahoo",
        "title": "HMM States Overlay on Price Data",
        "figsize": (16, 10),
        "dpi": 100,
        "state_colormap": "tab10",
        "state_alpha": 0.3,
        "show_volume": True,
        "show_grid": True,
        "mav": [],  # Moving averages
        "tight_layout": True,
        "returnfig": False,
    }

    if config:
        default_config.update(config)
    config = default_config

    # Input validation
    if not isinstance(price_data, pd.DataFrame):
        raise ValueError("price_data must be a pandas DataFrame")

    required_columns = ["open", "high", "low", "close"]
    missing_columns = [col for col in required_columns if col not in price_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in price_data: {missing_columns}")

    if len(states) != len(price_data):
        raise ValueError(
            f"Length mismatch: states ({len(states)}) != price_data ({len(price_data)})"
        )

    # Create color map for states
    unique_states = np.unique(states)
    n_states = len(unique_states)
    color_map = create_state_color_map(n_states, config["state_colormap"])

    logger.info(f"Visualizing {n_states} HMM states: {unique_states}")

    # Prepare additional plots
    addplot_list = []

    # Add state overlays as line plots on main panel
    for state_value in unique_states:
        if (
            state_value >= 0
        ):  # Skip negative states (typically represent missing/invalid data)
            state_mask = states == state_value
            if np.any(state_mask):
                color = color_map.get(state_value, "gray")

                # Create line plot for state prices
                state_prices = price_data["close"].copy()
                state_prices[~state_mask] = np.nan

                # Add as a line plot on the main panel
                addplot_list.append(
                    mpf.make_addplot(
                        state_prices,
                        type="line",
                        color=color,
                        alpha=config["state_alpha"],
                        panel=0,
                        secondary_y=False,
                    )
                )

    # Add technical indicators if provided
    if indicators is not None:
        config.get("indicators", {})

        # Common indicator configurations
        default_indicators = {
            "RSI_14": {"panel": 1, "color": "purple"},
            "MACD_12_26_9": {"panel": 2, "color": "blue"},
            "ATRr_14": {"panel": 3, "color": "orange"},
            "volume": {"panel": 4, "color": "green"},
        }

        for indicator_name, indicator_data in indicators.items():
            if indicator_name in default_indicators:
                ind_config = default_indicators[indicator_name]

                # Skip if indicator has all NaN values
                if indicator_data.notna().any():
                    addplot_list.append(
                        mpf.make_addplot(
                            indicator_data,
                            panel=ind_config["panel"],
                            color=ind_config["color"],
                            title=indicator_name,
                        )
                    )

    # Prepare plot arguments
    plot_kwargs = {
        "type": config["chart_type"],
        "style": config["style"],
        "title": config["title"],
        "figratio": (config["figsize"][0], config["figsize"][1]),
        "figscale": 1.0,
        "volume": config["show_volume"] and "volume" in price_data.columns,
        "mav": config["mav"],
        "tight_layout": config["tight_layout"],
        "returnfig": config["returnfig"],
    }

    # Only add addplot if it's not empty
    if addplot_list:
        plot_kwargs["addplot"] = addplot_list

    # Generate output path if not provided
    if output_path is None:
        output_path = "hmm_states_chart.png"

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Generate the plot
        logger.info(f"Generating chart: {config['title']}")

        # Set returnfig=True when we want to manipulate the figure
        plot_kwargs["returnfig"] = True
        result = mpf.plot(
            price_data,
            **plot_kwargs,
            savefig={"fname": output_path, "dpi": config["dpi"], "facecolor": "white"},
        )

        if result is not None:
            fig, axes = result
        else:
            fig, axes = None, None

        # Add legend for HMM states
        if fig and axes:
            # Create custom legend for states
            legend_elements = []
            for state_value in unique_states:
                if state_value >= 0:
                    color = color_map.get(state_value, "gray")
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=8,
                            label=f"State {state_value}",
                        )
                    )

            if legend_elements and len(axes) > 0:
                axes[0].legend(
                    handles=legend_elements, loc="upper left", framealpha=0.9
                )

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(
                output_path, dpi=config["dpi"], facecolor="white", bbox_inches="tight"
            )

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        logger.info(f"Chart saved successfully to: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to generate chart: {e}")
        raise


def plot_state_distribution(
    states: np.ndarray,
    indicators: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate distribution plots for indicators by HMM state.

    Args:
        states: HMM state sequence
        indicators: DataFrame with technical indicators
        config: Configuration dictionary
        output_path: Optional path to save the plot

    Returns:
        Path to the saved plot file
    """
    logger.info("Generating state distribution plots")

    # Default configuration
    default_config = {
        "figsize": (15, 10),
        "dpi": 100,
        "indicator_columns": None,  # Use all indicators if None
        "max_indicators": 6,  # Limit number of subplots
        "state_colormap": "tab10",
        "alpha": 0.7,
    }

    if config:
        default_config.update(config)
    config = default_config

    # Create state DataFrame
    pd.DataFrame({"state": states}, index=indicators.index)

    # Select indicators to plot
    if config["indicator_columns"] is None:
        indicator_columns = indicators.columns.tolist()
    else:
        indicator_columns = config["indicator_columns"]

    # Limit number of indicators
    if len(indicator_columns) > config["max_indicators"]:
        indicator_columns = indicator_columns[: config["max_indicators"]]

    # Create subplots
    n_indicators = len(indicator_columns)
    n_cols = min(3, n_indicators)
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=config["figsize"])
    if n_indicators == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Create color map
    unique_states = np.unique(states)
    unique_states = unique_states[unique_states >= 0]  # Skip negative states
    color_map = create_state_color_map(len(unique_states), config["state_colormap"])

    # Plot distributions
    for i, indicator in enumerate(indicator_columns):
        row = i // n_cols
        col = i % n_cols

        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        # Plot histogram for each state
        for j, state_value in enumerate(unique_states):
            state_mask = states == state_value
            state_data = indicators[indicator][state_mask].dropna()

            if len(state_data) > 0:
                color = color_map.get(j, "gray")
                ax.hist(
                    state_data,
                    bins=30,
                    alpha=config["alpha"],
                    color=color,
                    label=f"State {state_value}",
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_title(f"{indicator} Distribution by State")
        ax.set_xlabel(indicator)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_indicators, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)

    plt.tight_layout()

    # Generate output path if not provided
    if output_path is None:
        output_path = "hmm_state_distributions.png"

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(output_path, dpi=config["dpi"], facecolor="white", bbox_inches="tight")
    plt.close()

    logger.info(f"State distribution plot saved to: {output_path}")
    return str(output_path)


def create_regime_timeline_plot(
    states: np.ndarray,
    price_data: pd.Series,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Create a timeline plot showing HMM state transitions over time.

    Args:
        states: HMM state sequence
        price_data: Price series (typically close prices)
        config: Configuration dictionary
        output_path: Optional path to save the plot

    Returns:
        Path to the saved plot file
    """
    logger.info("Creating regime timeline plot")

    # Default configuration
    default_config = {
        "figsize": (16, 6),
        "dpi": 100,
        "title": "HMM Regime Timeline",
        "state_colormap": "tab10",
        "price_alpha": 0.7,
        "show_state_labels": True,
    }

    if config:
        default_config.update(config)
    config = default_config

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=config["figsize"], gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot price data on top panel
    ax1.plot(
        price_data.index,
        price_data.values,
        color="black",
        alpha=config["price_alpha"],
        linewidth=1,
    )
    ax1.set_title(config["title"])
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # Create state timeline on bottom panel
    unique_states = np.unique(states)
    unique_states = unique_states[unique_states >= 0]  # Skip negative states
    color_map = create_state_color_map(len(unique_states), config["state_colormap"])

    # Create colored regions for each state
    for i, state_value in enumerate(unique_states):
        state_mask = states == state_value
        color = color_map.get(i, "gray")

        # Plot state as colored regions
        ax2.fill_between(
            price_data.index,
            0,
            1,
            where=state_mask,
            alpha=0.7,
            color=color,
            label=f"State {state_value}",
        )

    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1.1)

    if config["show_state_labels"]:
        ax2.legend(loc="upper right")

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate output path if not provided
    if output_path is None:
        output_path = "hmm_regime_timeline.png"

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(output_path, dpi=config["dpi"], facecolor="white", bbox_inches="tight")
    plt.close()

    logger.info(f"Regime timeline plot saved to: {output_path}")
    return str(output_path)
