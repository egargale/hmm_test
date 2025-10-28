"""
HMM Futures Analysis CLI

Command-line interface for comprehensive HMM-based futures analysis regime detection system.
Provides end-to-end analysis from data loading to visualization and reporting.
"""

import signal
import sys
import time
import traceback
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.strategy_engine import StrategyEngine
from data_processing.csv_parser import process_csv
from data_processing.data_validation import validate_data
from data_processing.feature_engineering import add_features
from model_training.hmm_trainer import HMMTrainer
from model_training.inference_engine import StateInference
from processing_engines.factory import ProcessingEngineFactory
from utils import get_logger, setup_logging
from utils.data_types import BacktestConfig
from visualization.chart_generator import create_regime_timeline_plot, plot_states
from visualization.dashboard_builder import build_dashboard
from visualization.report_generator import generate_regime_report

# Global handler for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    shutdown_requested = True
    click.echo("\n🛑 Shutdown requested. Finishing current task...")
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@click.group()
@click.version_option(version="1.0.0", prog_name="hmm-analysis")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set logging level (default: INFO)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, log_level, quiet, verbose):
    """
    HMM Futures Analysis CLI

    A comprehensive command-line tool for Hidden Markov Model-based futures market analysis
    with regime detection, backtesting, and performance reporting.

    Features:
    - Multi-engine data processing (Streaming, Dask, Daft)
    - HMM training with automatic model selection
    - Regime-based backtesting with bias prevention
    - Interactive dashboards and detailed reports
    - Performance monitoring and progress tracking
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Set up logging
    if quiet:
        log_level = "ERROR"
    elif verbose:
        log_level = "DEBUG"

    setup_logging(level=log_level.upper())
    logger = get_logger(__name__)

    logger.info("HMM Futures Analysis CLI started")
    logger.debug(f"Log level: {log_level}")

    # Store global config in context
    ctx.obj["log_level"] = log_level
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose
    ctx.obj["logger"] = logger


@cli.command()
@click.option(
    "--input-csv",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file with futures data (OHLCV format)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for results (default: ./output)",
)
@click.option(
    "--n-states",
    "-n",
    type=click.IntRange(min=2, max=10),
    default=3,
    help="Number of HMM states (default: 3)",
)
@click.option(
    "--engine",
    type=click.Choice(["streaming", "dask", "daft"], case_sensitive=False),
    default="streaming",
    help="Processing engine to use (default: streaming)",
)
@click.option(
    "--target-column",
    type=str,
    default="close",
    help="Target column for HMM training (default: close)",
)
@click.option(
    "--test-size",
    type=click.FloatRange(min=0.1, max=0.5),
    default=0.2,
    help="Proportion of data for testing (default: 0.2)",
)
@click.option(
    "--lookahead-days",
    type=click.IntRange(min=0, max=10),
    default=1,
    help="Days for lookahead bias prevention (default: 1)",
)
@click.option(
    "--n-restarts",
    type=click.IntRange(min=1, max=20),
    default=10,
    help="Number of HMM training restarts (default: 10)",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option("--save-model", is_flag=True, default=True, help="Save trained HMM model")
@click.option(
    "--generate-charts",
    is_flag=True,
    default=True,
    help="Generate visualization charts",
)
@click.option(
    "--generate-dashboard",
    is_flag=True,
    default=True,
    help="Generate interactive dashboard",
)
@click.option(
    "--generate-report",
    is_flag=True,
    default=True,
    help="Generate detailed HTML report",
)
@click.pass_context
def analyze(
    ctx,
    input_csv,
    output_dir,
    n_states,
    engine,
    target_column,
    test_size,
    lookahead_days,
    n_restarts,
    random_seed,
    save_model,
    generate_charts,
    generate_dashboard,
    generate_report,
):
    """
    Run complete HMM analysis pipeline.

    This command executes the full analysis pipeline:
    1. Data loading and validation
    2. Feature engineering
    3. HMM training
    4. State inference
    5. Backtesting
    6. Performance analysis
    7. Visualization and reporting

    Example:
        hmm-analysis analyze -i data/futures.csv -o results/ -n 4 --engine dask
    """
    logger = ctx.obj["logger"]
    quiet = ctx.obj["quiet"]

    try:
        # Validate inputs
        logger.info("🚀 Starting HMM Futures Analysis Pipeline")
        logger.info(f"📂 Input file: {input_csv}")
        logger.info(f"📊 Output directory: {output_dir}")
        logger.info(f"🔢 Number of states: {n_states}")
        logger.info(f"⚙️  Processing engine: {engine}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Start timing
        start_time = time.time()

        # Step 1: Load and validate data
        logger.info("📁 Step 1: Loading and validating data...")

        if not quiet:
            click.echo("Loading data...")

        try:
            data = process_csv(str(input_csv))
            validation_result = validate_data(data)

            if not validation_result["is_valid"]:
                raise ValueError(
                    f"Data validation failed: {validation_result['errors']}"
                )

            logger.info(f"✅ Loaded {len(data)} rows of data")

        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise click.ClickException(f"Failed to load data: {e}") from e

        # Step 2: Feature engineering
        logger.info("⚙️  Step 2: Feature engineering...")

        if not quiet:
            click.echo("Engineering features...")

        try:
            # Initialize processing engine
            processing_engine = ProcessingEngineFactory.create_engine(engine)
            logger.info(f"Using {engine} processing engine")

            # Initialize feature engineering configuration
            indicator_config = {
                "returns": {"periods": [1, 5, 10]},
                "moving_averages": {"periods": [5, 10, 20]},
                "volatility": {"periods": [14]},
                "momentum": {"periods": [14]},
                "volume": {"enabled": True},
            }

            # Apply feature engineering using selected engine
            if engine == "dask":
                import dask.dataframe as dd

                ddf = dd.from_pandas(data, npartitions=4)

                with tqdm(total=4, desc="Processing features", disable=quiet) as pbar:
                    # Apply feature engineering to Dask DataFrame
                    def apply_features(df):
                        return add_features(df, config=indicator_config)

                    features_dd = processing_engine.process(ddf, apply_features)
                    pbar.update(1)

                    features = features_dd.compute()
                    pbar.update(3)

            elif engine == "daft":
                import daft

                df = daft.from_pandas(data)

                with tqdm(total=2, desc="Processing features", disable=quiet) as pbar:
                    # Apply feature engineering to Daft DataFrame
                    def apply_features(df):
                        return add_features(df, config=indicator_config)

                    features_df = processing_engine.process(df, apply_features)
                    pbar.update(1)

                    features = features_df.to_pandas()
                    pbar.update(1)

            else:  # streaming
                with tqdm(total=100, desc="Processing features", disable=quiet) as pbar:
                    # Apply feature engineering to pandas DataFrame
                    def apply_features(df):
                        return add_features(df, config=indicator_config)

                    features = processing_engine.process(
                        data, apply_features, progress_callback=lambda x: pbar.update(x)
                    )

            # Handle NaN values in features
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

            logger.info(
                f"✅ Feature engineering completed. Generated {len(features.columns)} features"
            )

        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            raise click.ClickException(f"Feature engineering failed: {e}") from e

        # Step 3: HMM Training
        logger.info("🧠 Step 3: HMM model training...")

        if not quiet:
            click.echo(f"Training HMM with {n_states} states...")

        try:
            trainer = HMMTrainer(
                n_states=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=random_seed,
                tol=1e-4,
            )

            # Prepare training data
            train_size = int(len(features) * (1 - test_size))
            train_data = features[target_column].iloc[:train_size].values

            # Train model with progress tracking
            with tqdm(total=n_restarts, desc="Training models", disable=quiet) as pbar:
                model, metadata = trainer.train_with_restarts(
                    train_data,
                    n_restarts=n_restarts,
                    progress_callback=lambda x: pbar.update(1),
                )

            logger.info(
                f"✅ HMM training completed. Best log-likelihood: {metadata['log_likelihood']:.2f}"
            )

            # Save model if requested
            if save_model:
                model_path = output_dir / f"hmm_model_{n_states}states.pkl"
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump({"model": model, "metadata": metadata}, f)
                logger.info(f"💾 Model saved to {model_path}")

        except Exception as e:
            logger.error(f"❌ HMM training failed: {e}")
            raise click.ClickException(f"HMM training failed: {e}") from e

        # Step 4: State inference
        logger.info("🔍 Step 4: State inference...")

        if not quiet:
            click.echo("Inferring hidden states...")

        try:
            inference = StateInference(model)

            # Infer states for full dataset
            with tqdm(total=100, desc="Inferring states", disable=quiet) as pbar:
                states = inference.infer_states(
                    features[target_column].values,
                    progress_callback=lambda x: pbar.update(x),
                )

            logger.info(
                f"✅ State inference completed. Found {len(np.unique(states))} unique states"
            )

        except Exception as e:
            logger.error(f"❌ State inference failed: {e}")
            raise click.ClickException(f"State inference failed: {e}") from e

        # Step 5: Backtesting
        logger.info("💰 Step 5: Regime-based backtesting...")

        if not quiet:
            click.echo("Running backtesting simulation...")

        try:
            # Create backtest configuration
            backtest_config = BacktestConfig(
                initial_capital=100000.0,
                commission=0.001,  # 0.1%
                slippage=0.0001,  # 0.01%
                lookahead_bias_prevention=True,
                lookahead_days=lookahead_days,
            )

            # Initialize strategy engine
            strategy_engine = StrategyEngine(backtest_config)

            # Create state-to-position mapping
            unique_states = np.unique(states)
            state_mapping = {}
            for i, state in enumerate(unique_states):
                if state < 0:
                    state_mapping[state] = 0  # Neutral for invalid states
                else:
                    # Simple mapping: even states = long, odd states = short
                    state_mapping[state] = 1 if i % 2 == 0 else -1

            # Run backtesting
            with tqdm(total=100, desc="Backtesting", disable=quiet) as pbar:
                backtest_result = strategy_engine.backtest_strategy(
                    data=data,
                    states=states,
                    state_mapping=state_mapping,
                    progress_callback=lambda x: pbar.update(x),
                )

            logger.info(
                f"✅ Backtesting completed. Generated {len(backtest_result.trades)} trades"
            )

        except Exception as e:
            logger.error(f"❌ Backtesting failed: {e}")
            raise click.ClickException(f"Backtesting failed: {e}") from e

        # Step 6: Performance analysis
        logger.info("📈 Step 6: Performance analysis...")

        if not quiet:
            click.echo("Analyzing performance...")

        try:
            analyzer = PerformanceAnalyzer()

            with tqdm(total=100, desc="Computing metrics", disable=quiet) as pbar:
                metrics = analyzer.calculate_performance(
                    backtest_result.equity_curve,
                    backtest_result.positions,
                    benchmark=data["close"].pct_change(),
                    progress_callback=lambda x: pbar.update(x),
                )

            logger.info(
                f"✅ Performance analysis completed. Sharpe ratio: {metrics.sharpe_ratio:.2f}"
            )

        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            raise click.ClickException(f"Performance analysis failed: {e}") from e

        # Step 7: Visualization and reporting
        logger.info("📊 Step 7: Generating visualizations and reports...")

        try:
            if generate_charts:
                if not quiet:
                    click.echo("Generating charts...")

                with tqdm(total=3, desc="Creating charts", disable=quiet) as pbar:
                    # Main state visualization
                    chart_path = output_dir / "hmm_states_chart.png"
                    plot_states(
                        price_data=data,
                        states=states,
                        indicators=features,
                        output_path=str(chart_path),
                        show_plot=False,
                    )
                    pbar.update(1)

                    # Regime timeline
                    timeline_path = output_dir / "regime_timeline.png"
                    create_regime_timeline_plot(
                        states=states,
                        price_data=data["close"],
                        output_path=str(timeline_path),
                    )
                    pbar.update(1)

                    # State distributions
                    from visualization.chart_generator import plot_state_distribution

                    dist_path = output_dir / "state_distributions.png"
                    plot_state_distribution(
                        states=states, indicators=features, output_path=str(dist_path)
                    )
                    pbar.update(1)

                logger.info("✅ Charts generated successfully")

            if generate_dashboard:
                if not quiet:
                    click.echo("Generating dashboard...")

                with tqdm(total=100, desc="Creating dashboard", disable=quiet) as pbar:
                    dashboard_path = output_dir / "dashboard.html"
                    build_dashboard(
                        result=backtest_result,
                        metrics=metrics,
                        states=states,
                        progress_callback=lambda x: pbar.update(x),
                        output_path=str(dashboard_path),
                    )

                logger.info("✅ Dashboard generated successfully")

            if generate_report:
                if not quiet:
                    click.echo("Generating report...")

                with tqdm(total=100, desc="Creating report", disable=quiet) as pbar:
                    report_path = output_dir / "analysis_report.html"
                    generate_regime_report(
                        result=backtest_result,
                        metrics=metrics,
                        states=states,
                        indicators=features,
                        progress_callback=lambda x: pbar.update(x),
                        output_path=str(report_path),
                    )

                logger.info("✅ Report generated successfully")

        except Exception as e:
            logger.error(f"❌ Visualization/reporting failed: {e}")
            # Don't raise ClickException for visualization errors - just log them

        # Calculate total execution time
        total_time = time.time() - start_time

        # Final summary
        logger.info("🎉 HMM Analysis Pipeline Completed Successfully!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")

        if not quiet:
            click.echo("\n" + "=" * 60)
            click.echo("🎉 HMM ANALYSIS COMPLETED SUCCESSFULLY!")
            click.echo("=" * 60)
            click.echo(f"📂 Results saved to: {output_dir}")
            click.echo(f"📊 Data processed: {len(data)} rows")
            click.echo(f"🔢 HMM states: {n_states}")
            click.echo(f"💰 Trades generated: {len(backtest_result.trades)}")
            click.echo(f"📈 Sharpe ratio: {metrics.sharpe_ratio:.2f}")
            click.echo(f"⏱️  Total time: {total_time:.2f} seconds")

            if generate_charts:
                click.echo(f"📊 Charts: {output_dir}/*.png")
            if generate_dashboard:
                click.echo(f"🖥️  Dashboard: {output_dir}/dashboard.html")
            if generate_report:
                click.echo(f"📄 Report: {output_dir}/analysis_report.html")
            if save_model:
                click.echo(f"💾 Model: {output_dir}/hmm_model_{n_states}states.pkl")

    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed: {e}")
        if not quiet:
            click.echo(f"\n❌ Error: {e}", err=True)
            if ctx.obj["verbose"]:
                click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(f"Analysis failed: {e}") from e


@cli.command()
@click.option(
    "--input-csv",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file to validate",
)
def validate(input_csv):
    """
    Validate input data format and structure.

    Checks for required OHLCV columns, data types, and common issues.
    """

    try:
        click.echo(f"Validating {input_csv}...")

        # Load and validate data
        data = process_csv(str(input_csv))
        validation_result = validate_data(data)

        if validation_result["is_valid"]:
            click.echo("✅ Data validation passed!")
            click.echo(f"📊 {len(data)} rows of data")
            click.echo(f"📅 Date range: {data.index.min()} to {data.index.max()}")
            click.echo(f"📈 Columns: {list(data.columns)}")
        else:
            click.echo("❌ Data validation failed:", err=True)
            for error in validation_result["errors"]:
                click.echo(f"  • {error}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    click.echo("HMM Futures Analysis CLI v1.0.0")
    click.echo("© 2024 - Advanced Regime Detection System")


if __name__ == "__main__":
    cli()
