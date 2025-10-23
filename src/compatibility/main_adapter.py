"""
Backward compatibility adapter for main.py functionality.

This module provides a compatibility layer that allows the new pipeline
to work with the original main.py CLI interface, ensuring smooth
migration without breaking existing workflows.
"""

import argparse
import asyncio
import logging
import warnings
from pathlib import Path

from ..pipelines.hmm_pipeline import HMMPipeline
from ..utils.logging_config import setup_logger

# Set up deprecation warning
logger = setup_logger(__name__)


def emit_deprecation_warning():
    """Emit deprecation warning for main.py usage"""
    warnings.warn(
        "Using main.py directly is deprecated. Please migrate to the new CLI interface. "
        "Run 'python -m src.cli.hmm_commands --help' for the new interface.",
        DeprecationWarning,
        stacklevel=3
    )


def create_legacy_parser() -> argparse.ArgumentParser:
    """
    Create argument parser that matches main.py exactly.

    Returns:
        ArgumentParser configured for main.py compatibility
    """
    parser = argparse.ArgumentParser(
        description="Train HMM on huge futures CSV (Legacy Interface - Deprecated)"
    )

    # Core arguments (exact match with main.py)
    parser.add_argument("csv", help="Path to futures OHLCV CSV")
    parser.add_argument(
        "-n",
        "--n_states",
        type=int,
        default=3,
        help="Number of hidden states (default 3)",
    )
    parser.add_argument(
        "-i",
        "--max_iter",
        type=int,
        default=100,
        help="Max EM iterations (default 100)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Save quick sanity plot",
    )
    parser.add_argument(
        "--model-path",
        help="Path to pre-trained model and scaler",
    )
    parser.add_argument(
        "--model-out",
        help="Path to save trained model and scaler",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for reading CSV (default 100000)",
    )
    parser.add_argument(
        "--prevent-lookahead",
        action="store_true",
        help="Prevent lookahead bias by shifting positions",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run simple backtest after training",
    )

    return parser


def main(args=None):
    """
    Main function that replicates main.py behavior using new pipeline.

    Args:
        args: Command line arguments (None for sys.argv)
    """
    # Emit deprecation warning
    emit_deprecation_warning()

    # Parse arguments
    parser = create_legacy_parser()
    args = parser.parse_args(args)

    # Set up logging to match main.py
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Convert to async main
    return asyncio.run(async_main(args))


async def async_main(args):
    """
    Async main function using the new pipeline.

    Args:
        args: Parsed command line arguments
    """
    try:
        # Validate arguments (same as main.py)
        if args.n_states < 1:
            raise ValueError("Number of states must be positive")
        if args.max_iter < 1:
            raise ValueError("Max iterations must be positive")
        if args.chunksize < 1:
            raise ValueError("Chunk size must be positive")

        # Validate CSV file exists
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Create pipeline from legacy args
        pipeline = HMMPipeline.from_args(args)

        # Run pipeline
        logger.info("Starting HMM analysis using legacy interface...")
        result = await pipeline.run(csv_path)

        # Handle legacy plotting
        if args.plot and result.processed_data is not None:
            await _generate_legacy_plot(result.processed_data, args.n_states, csv_path)

        # Log results in legacy format
        _log_legacy_results(result, args)

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


async def _generate_legacy_plot(data, n_states, csv_path):
    """
    Generate plot that matches main.py output format.

    Args:
        data: Processed dataframe with states
        n_states: Number of HMM states
        csv_path: Original CSV path for output naming
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 4))
        plt.plot(data.index, data["Close"], label="Close")

        for s in range(n_states):
            mask = data["state"] == s
            plt.scatter(
                data.index[mask],
                data["Close"][mask],
                label=f"State {s}",
                s=5,
            )

        plt.legend()
        plt.title("Futures Close Price & HMM States")
        plt.tight_layout()

        # Save with same naming convention as main.py
        plot_path = csv_path.with_suffix(".png")
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Plot saved to {plot_path}")

    except ImportError:
        logger.warning("matplotlib not found – skipping plot.")
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
        # Don't fail execution for plot issues


def _log_legacy_results(result, args):
    """
    Log results in the same format as main.py.

    Args:
        result: Pipeline execution result
        args: Original command line arguments
    """
    # Log training results if available
    if result.model:
        # Try to get model information
        try:
            converged = getattr(result.model, 'monitor_', None)
            if converged and hasattr(converged, 'converged'):
                logger.info(f"Model converged: {converged.converged}")

            if hasattr(result.model, 'score_'):
                logger.info(f"Log-likelihood: {result.model.score_:.2f}")
        except Exception:
            pass  # Don't fail for logging issues

    # Log backtesting results if available
    if args.backtest and result.performance_metrics:
        perf = result.performance_metrics

        # Map new metrics to legacy format
        if 'equity_curve' in perf:
            logger.info(f"Final Equity: {perf['equity_curve']:.4f}")

        if 'sharpe_ratio' in perf:
            logger.info(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")

        if 'max_drawdown' in perf:
            logger.info(f"Max Drawdown: {perf['max_drawdown']:.4f}")


# Legacy function mappings for exact compatibility
def add_features(df):
    """
    Legacy wrapper for feature engineering.

    DEPRECATED: Use FeatureEngineer.add_features() instead.
    """
    warnings.warn(
        "add_features() is deprecated. Use FeatureEngineer.add_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from ..data_processing.feature_engineering import FeatureEngineer
    from ..pipelines.pipeline_types import FeatureConfig

    engineer = FeatureEngineer(FeatureConfig())
    return engineer.add_features(df)


def stream_features(csv_path, chunksize=100_000):
    """
    Legacy wrapper for streaming feature processing.

    DEPRECATED: Use StreamingDataProcessor.process_stream() instead.
    """
    warnings.warn(
        "stream_features() is deprecated. Use StreamingDataProcessor.process_stream() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from ..data_processing.streaming_processor import StreamingDataProcessor
    from ..data_processing.feature_engineering import FeatureEngineer
    from ..pipelines.pipeline_types import (
        FeatureConfig, StreamingConfig, ProcessingMode
    )

    # Create configuration
    features = FeatureConfig()
    streaming = StreamingConfig(
        chunk_size=chunksize,
        processing_mode=ProcessingMode.STREAMING
    )

    # Create processor and engineer
    processor = StreamingDataProcessor(streaming)
    engineer = FeatureEngineer(features)

    # Run synchronously for compatibility
    import asyncio
    return asyncio.run(processor.process_stream(csv_path, engineer))


def simple_backtest(df, states):
    """
    Legacy wrapper for simple backtesting.

    DEPRECATED: Use StrategyEngine.generate_positions() instead.
    """
    warnings.warn(
        "simple_backtest() is deprecated. Use StrategyEngine.generate_positions() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from ..backtesting.strategy_engine import StrategyEngine
    from ..pipelines.pipeline_types import BacktestConfig

    # Create simple strategy config
    config = BacktestConfig(
        strategy_type="state_based",
        long_states=[0],  # long low-vol up
        short_states=[2]  # short high-vol down
    )

    engine = StrategyEngine(config)
    positions = engine.generate_positions(df, states)
    returns = engine.calculate_returns(df, positions)

    return returns.cumsum()


def perf_metrics(series):
    """
    Legacy wrapper for performance metrics.

    DEPRECATED: Use PerformanceAnalyzer.analyze() instead.
    """
    warnings.warn(
        "perf_metrics() is deprecated. Use PerformanceAnalyzer.analyze() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from ..backtesting.performance_analyzer import PerformanceAnalyzer

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(series)

    # Map to legacy format
    returns = series.diff().dropna()
    sharpe = returns.mean() / returns.std() * (252 * 78) ** 0.5  # Annualized
    drawdown = (series - series.cummax()).min()

    return sharpe, drawdown


if __name__ == "__main__":
    main()