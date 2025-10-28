"""
HMM Futures Analysis CLI - Comprehensive Orchestration

A comprehensive command-line interface for HMM futures market analysis
with full orchestration, error handling, progress monitoring, and memory management.
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import numpy as np
from tqdm import tqdm

# Optional imports
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import utilities
from data_processing.data_validation import validate_data
from data_processing.feature_engineering import add_features
from model_training import inference_engine
from model_training.hmm_trainer import train_model, validate_features_for_hmm
from model_training.model_persistence import load_model, save_model
from processing_engines.index import ProcessingEngineFactory
from utils import get_logger, setup_logging

# Global logger and configuration
logger = None
current_memory_usage = 0.0
MEMORY_WARNING_THRESHOLD = 0.8  # 80% of available RAM


def get_memory_usage() -> float:
    """Get current memory usage as percentage of available RAM."""
    if not PSUTIL_AVAILABLE:
        return 0.0

    try:
        process = psutil.Process()
        memory_percent = process.memory_percent()
        return memory_percent / 100.0
    except Exception:
        return 0.0


def check_memory_usage(operation: str = "operation"):
    """Check memory usage and log warnings if threshold exceeded."""
    if not PSUTIL_AVAILABLE:
        return

    global current_memory_usage
    current_memory_usage = get_memory_usage()

    if current_memory_usage > MEMORY_WARNING_THRESHOLD:
        logger.warning(
            f"High memory usage during {operation}: {current_memory_usage:.1%} of available RAM"
        )
        # Trigger garbage collection
        gc.collect()
        # Re-check after collection
        new_usage = get_memory_usage()
        if new_usage < current_memory_usage:
            logger.info(f"Memory reduced after garbage collection: {new_usage:.1%}")


def log_performance_metrics(
    start_time: float, operation: str, additional_info: Dict[str, Any] = None
):
    """Log performance metrics for completed operations."""
    elapsed_time = time.time() - start_time
    memory_usage = get_memory_usage()

    metrics = {
        "operation": operation,
        "elapsed_time_seconds": elapsed_time,
        "memory_usage_percent": memory_usage,
        "timestamp": time.time(),
    }

    if additional_info:
        metrics.update(additional_info)

    memory_info = f", Memory: {memory_usage:.1%}" if PSUTIL_AVAILABLE else ""
    logger.info(f"Performance - {operation}: {elapsed_time:.2f}s{memory_info}")

    return metrics


class HMMConfig:
    """Configuration class for HMM analysis parameters."""

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        tol: float = 1e-3,
        num_restarts: int = 3,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.num_restarts = num_restarts

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "tol": self.tol,
            "num_restarts": self.num_restarts,
        }


class ProcessingConfig:
    """Configuration class for data processing parameters."""

    def __init__(
        self,
        engine_type: str = "streaming",
        chunk_size: int = 100000,
        indicators: Optional[Dict[str, Any]] = None,
    ):
        self.engine_type = engine_type
        self.chunk_size = chunk_size
        self.indicators = indicators or {
            "sma_5": {"window": 5},
            "sma_10": {"window": 10},
            "sma_20": {"window": 20},
            "volatility_14": {"window": 14},
            "returns": {},
        }


@click.group()
@click.version_option(version="1.0.0", prog_name="hmm-futures-analysis")
@click.option(
    "--config-file", type=click.Path(exists=True), help="Configuration file (JSON/YAML)"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set logging level (default: INFO)",
)
@click.option(
    "--memory-monitor/--no-memory-monitor",
    default=True,
    help="Enable memory monitoring (default: enabled)",
)
@click.pass_context
def cli(ctx, config_file, log_level, memory_monitor):
    """
    HMM Futures Analysis CLI - Comprehensive Orchestration

    A production-ready command-line tool for comprehensive HMM futures market analysis
    with multi-engine processing, advanced error handling, and performance monitoring.
    """
    # Set up logging
    setup_logging(level=log_level.upper())
    global logger
    logger = get_logger(__name__)

    logger.info("🚀 HMM Futures Analysis CLI started")
    logger.info(f"📊 Log level: {log_level}")
    logger.info(f"🧠 Memory monitoring: {'enabled' if memory_monitor else 'disabled'}")

    # Load configuration if provided
    config = {}
    if config_file:
        try:
            with open(config_file) as f:
                if config_file.endswith(".json"):
                    config = json.load(f)
                else:
                    # Simple YAML parsing (basic)
                    if YAML_AVAILABLE:
                        config = yaml.safe_load(f)
                    else:
                        raise click.ClickException(
                            "YAML support requires PyYAML package"
                        )
            logger.info(f"✅ Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise click.ClickException(f"Configuration loading failed: {e}") from e

    # Store global config in context
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["logger"] = logger
    ctx.obj["config"] = config
    ctx.obj["memory_monitor"] = memory_monitor

    # Initial memory check
    if memory_monitor:
        check_memory_usage("CLI initialization")


@cli.command()
@click.option(
    "--input-csv",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file with futures data",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for results",
)
@click.option(
    "--engine",
    type=click.Choice(["streaming", "dask", "daft"], case_sensitive=False),
    default="streaming",
    help="Processing engine (default: streaming)",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100000,
    help="Chunk size for processing (default: 100000)",
)
@click.pass_context
def validate(ctx, input_csv, output_dir, engine, chunk_size):
    """
    Validate input data format and structure using specified processing engine.
    """
    logger = ctx.obj["logger"]
    memory_monitor = ctx.obj["memory_monitor"]

    try:
        click.echo(f"🔍 Validating {input_csv} using {engine} engine...")
        start_time = time.time()

        # Create processing config
        proc_config = ProcessingConfig(engine_type=engine, chunk_size=chunk_size)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check memory usage
        if memory_monitor:
            check_memory_usage("data validation")

        # Load and validate data using processing engine
        logger.info(f"📁 Loading data with {engine} engine...")

        try:
            # Get processing engine
            process_func = ProcessingEngineFactory().get_engine(engine)

            # Process data
            with tqdm(total=1, desc="Processing data") as pbar:
                data = process_func(str(input_csv), proc_config)
                pbar.update(1)

            if memory_monitor:
                check_memory_usage("data loading")

            # Validate processed data
            logger.info("✅ Running data validation...")
            data_clean, validation_result = validate_data(data)

            # Check if validation succeeded (no critical issues)
            critical_issues = [
                issue
                for issue in validation_result["issues_found"]
                if issue.get("severity") == "critical"
            ]

            if not critical_issues:
                click.echo("✅ Data validation passed!")
                click.echo(f"📊 {len(data_clean)} rows of data")
                click.echo(
                    f"📅 Date range: {data_clean.index.min()} to {data_clean.index.max()}"
                )
                click.echo(f"📈 Columns: {list(data_clean.columns)}")

                # Save validation report
                report_path = output_dir / "validation_report.txt"
                with open(report_path, "w") as f:
                    f.write("Data Validation Report\n")
                    f.write("======================\n\n")
                    f.write(f"File: {input_csv}\n")
                    f.write(f"Engine: {engine}\n")
                    f.write(f"Chunk size: {chunk_size}\n")
                    f.write(f"Rows: {len(data_clean)}\n")
                    f.write(f"Columns: {list(data_clean.columns)}\n")
                    f.write(
                        f"Date range: {data_clean.index.min()} to {data_clean.index.max()}\n"
                    )
                    f.write(
                        f"Quality score: {validation_result.get('quality_score', 'N/A')}\n"
                    )
                    f.write(f"Issues found: {len(validation_result['issues_found'])}\n")
                    f.write(f"Critical issues: {len(critical_issues)}\n")
                    f.write("\nValidation status: PASSED\n")

                click.echo(f"📄 Validation report saved to: {report_path}")

                # Log performance metrics
                log_performance_metrics(
                    start_time,
                    "data_validation",
                    {"rows_processed": len(data_clean), "engine": engine},
                )

            else:
                click.echo("❌ Data validation failed:", err=True)
                for issue in critical_issues:
                    click.echo(
                        f"  • {issue.get('description', 'Unknown error')}", err=True
                    )
                sys.exit(1)

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            click.echo(f"❌ Validation failed: {e}", err=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Validation command failed: {e}")
        raise click.ClickException(f"Validation failed: {e}") from e


@cli.command()
@click.option(
    "--input-csv",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file with futures data",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for results",
)
@click.option(
    "--engine",
    type=click.Choice(["streaming", "dask", "daft"], case_sensitive=False),
    default="streaming",
    help="Processing engine (default: streaming)",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100000,
    help="Chunk size for processing (default: 100000)",
)
@click.option(
    "--n-states",
    "-n",
    type=click.IntRange(min=2, max=10),
    default=3,
    help="Number of HMM states (default: 3)",
)
@click.option(
    "--test-size",
    type=click.FloatRange(min=0.1, max=0.5),
    default=0.2,
    help="Proportion of data for testing (default: 0.2)",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "--covariance-type",
    type=click.Choice(["full", "diag", "spherical", "tied"]),
    default="full",
    help="HMM covariance type (default: full)",
)
@click.option(
    "--max-iter", type=int, default=100, help="Maximum HMM iterations (default: 100)"
)
@click.option(
    "--num-restarts",
    type=int,
    default=3,
    help="Number of HMM restarts for best model (default: 3)",
)
@click.option(
    "--model-out", type=click.Path(path_type=Path), help="Path to save trained model"
)
@click.option(
    "--generate-charts/--no-generate-charts",
    default=False,
    help="Generate visualization charts",
)
@click.option(
    "--generate-dashboard/--no-generate-dashboard",
    default=False,
    help="Generate interactive dashboard",
)
@click.option(
    "--generate-report/--no-generate-report",
    default=False,
    help="Generate detailed report",
)
@click.pass_context
def analyze(
    ctx,
    input_csv,
    output_dir,
    engine,
    chunk_size,
    n_states,
    test_size,
    random_seed,
    covariance_type,
    max_iter,
    num_restarts,
    model_out,
    generate_charts,
    generate_dashboard,
    generate_report,
):
    """
    Run comprehensive HMM analysis pipeline with full orchestration.

    This command executes a complete analysis pipeline:
    1. Data loading and validation using specified engine
    2. Advanced feature engineering
    3. HMM model training with multiple restarts
    4. State inference and analysis
    5. Optional visualization and reporting
    """
    logger = ctx.obj["logger"]
    memory_monitor = ctx.obj["memory_monitor"]

    try:
        # Validate inputs and log configuration
        logger.info("🚀 Starting Comprehensive HMM Analysis Pipeline")
        click.echo(f"📂 Input file: {input_csv}")
        click.echo(f"📊 Output directory: {output_dir}")
        click.echo(f"⚙️  Processing engine: {engine}")
        click.echo(f"🔢 Number of states: {n_states}")
        click.echo(f"🎲 Random seed: {random_seed}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize configurations
        processing_config = ProcessingConfig(engine_type=engine, chunk_size=chunk_size)

        hmm_config = HMMConfig(
            n_states=n_states,
            covariance_type=covariance_type,
            n_iter=max_iter,
            random_state=random_seed,
            num_restarts=num_restarts,
        )

        # Start timing
        pipeline_start_time = time.time()

        # Step 1: Data Processing
        logger.info("📁 Step 1: Data Loading and Processing...")
        click.echo("Loading and processing data...")

        step_start_time = time.time()
        try:
            # Get processing engine
            process_func = ProcessingEngineFactory().get_engine(engine)

            # Process data with progress bar
            with tqdm(total=1, desc="Processing data") as pbar:
                data = process_func(str(input_csv), processing_config)
                pbar.update(1)

            if memory_monitor:
                check_memory_usage("data processing")

            # Validate processed data
            data_clean, validation_result = validate_data(data)

            # Check for critical validation issues
            critical_issues = [
                issue
                for issue in validation_result["issues_found"]
                if issue.get("severity") == "critical"
            ]

            if critical_issues:
                error_descriptions = [
                    issue.get("description", "Unknown error")
                    for issue in critical_issues
                ]
                raise ValueError(f"Data validation failed: {error_descriptions}")

            logger.info(f"✅ Loaded and validated {len(data_clean)} rows of data")

            log_performance_metrics(
                step_start_time,
                "data_processing",
                {"rows_processed": len(data_clean), "engine": engine},
            )

        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            raise click.ClickException(f"Data processing failed: {e}") from e

        # Step 2: Feature Engineering
        logger.info("⚙️  Step 2: Advanced Feature Engineering...")
        click.echo("Engineering features...")

        step_start_time = time.time()
        try:
            # Use cleaned data from validation
            features = add_features(data_clean)

            # Drop NaN values created by rolling windows
            features_clean = features.dropna()

            logger.info("✅ Feature engineering completed")
            logger.info(f"   Original features: {len(data_clean.columns)}")
            logger.info(f"   Engineered features: {len(features.columns)}")
            logger.info(f"   Final dataset: {len(features_clean)} rows")

            if memory_monitor:
                check_memory_usage("feature engineering")

            log_performance_metrics(
                step_start_time,
                "feature_engineering",
                {
                    "original_features": len(data_clean.columns),
                    "engineered_features": len(features.columns),
                    "final_rows": len(features_clean),
                },
            )

        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            raise click.ClickException(f"Feature engineering failed: {e}") from e

        # Step 3: HMM Model Training
        logger.info("🧠 Step 3: HMM Model Training...")
        click.echo(f"Training HMM with {n_states} states...")

        step_start_time = time.time()
        try:
            # Prepare training data
            train_size = int(len(features_clean) * (1 - test_size))
            train_data = features_clean.iloc[:train_size]
            test_data = features_clean.iloc[train_size:]

            # Use close prices for HMM (can be extended to use multiple features)
            X_train = train_data["close"].values.reshape(-1, 1)
            X_test = test_data["close"].values.reshape(-1, 1)

            # Validate features
            validate_features_for_hmm(X_train)

            # Train model with progress indication
            logger.info(f"Training HMM model with {num_restarts} restarts...")

            model, metadata = train_model(X_train, hmm_config.to_dict())

            logger.info("✅ HMM training completed")
            logger.info(f"   Number of states: {metadata['n_components']}")
            logger.info(f"   Convergence: {metadata['converged']}")
            logger.info(f"   Log-likelihood: {metadata['log_likelihood']:.2f}")
            logger.info(f"   Training samples: {metadata['n_samples']}")

            if memory_monitor:
                check_memory_usage("hmm_training")

            log_performance_metrics(
                step_start_time,
                "hmm_training",
                {
                    "n_states": metadata["n_components"],
                    "converged": metadata["converged"],
                    "log_likelihood": metadata["log_likelihood"],
                    "training_samples": metadata["n_samples"],
                },
            )

        except Exception as e:
            logger.error(f"❌ HMM training failed: {e}")
            raise click.ClickException(f"HMM training failed: {e}") from e

        # Step 4: State Inference
        logger.info("🔍 Step 4: State Inference...")
        click.echo("Inferring hidden states...")

        step_start_time = time.time()
        try:
            # Infer states for training and test data
            train_states = inference_engine.predict_states(model, X_train)
            test_states = inference_engine.predict_states(model, X_test)

            # Combine states
            all_states = np.concatenate([train_states, test_states])

            logger.info("✅ State inference completed")
            logger.info(f"   Training states: {len(train_states)}")
            logger.info(f"   Test states: {len(test_states)}")
            logger.info(f"   Unique states: {len(np.unique(all_states))}")

            if memory_monitor:
                check_memory_usage("state_inference")

            log_performance_metrics(
                step_start_time,
                "state_inference",
                {
                    "training_states": len(train_states),
                    "test_states": len(test_states),
                    "unique_states": len(np.unique(all_states)),
                },
            )

        except Exception as e:
            logger.error(f"❌ State inference failed: {e}")
            raise click.ClickException(f"State inference failed: {e}") from e

        # Step 5: Results Analysis and Saving
        logger.info("💾 Step 5: Results Analysis and Saving...")
        click.echo("Analyzing and saving results...")

        step_start_time = time.time()
        try:
            # Create results DataFrame
            results = features_clean.copy()
            results["hmm_state"] = all_states

            # Calculate state statistics
            state_stats = {}
            for state in range(n_states):
                state_mask = results["hmm_state"] == state
                if state_mask.sum() > 0:
                    state_returns = results.loc[state_mask, "returns"]
                    state_stats[state] = {
                        "count": state_mask.sum(),
                        "percentage": state_mask.sum() / len(results) * 100,
                        "mean_return": state_returns.mean(),
                        "std_return": state_returns.std(),
                        "volatility": results.loc[state_mask, "volatility_14"].mean(),
                        "sharpe_ratio": (
                            state_returns.mean() / state_returns.std() * np.sqrt(252)
                            if state_returns.std() > 0
                            else 0
                        ),
                    }

            # Save states
            states_path = output_dir / "states.csv"
            results.to_csv(states_path)
            logger.info(f"✅ States saved to {states_path}")

            # Save model and metadata
            if model_out:
                model_path = Path(model_out)
            else:
                model_path = output_dir / "hmm_model.pkl"

            save_model(model, None, hmm_config.to_dict(), str(model_path))
            logger.info(f"✅ Model saved to {model_path}")

            # Save model information
            model_info_path = output_dir / "model_info.txt"
            with open(model_info_path, "w") as f:
                f.write("HMM Model Information\n")
                f.write("=====================\n\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\nConfiguration:\n")
                for key, value in hmm_config.to_dict().items():
                    f.write(f"  {key}: {value}\n")

            # Save state statistics
            stats_path = output_dir / "state_statistics.txt"
            with open(stats_path, "w") as f:
                f.write("State Statistics\n")
                f.write("================\n\n")
                for state, stats in state_stats.items():
                    f.write(f"State {state}:\n")
                    for key, value in stats.items():
                        f.write(f"  {key}: {value:.4f}\n")
                    f.write("\n")

            logger.info("✅ Analysis results saved")

            if memory_monitor:
                check_memory_usage("results_saving")

            log_performance_metrics(
                step_start_time,
                "results_saving",
                {
                    "states_saved": len(results),
                    "unique_states": len(np.unique(all_states)),
                },
            )

        except Exception as e:
            logger.error(f"❌ Results saving failed: {e}")
            # Don't fail the entire pipeline for saving issues

        # Step 6: Optional Visualization and Reporting
        if generate_charts or generate_dashboard or generate_report:
            logger.info("📊 Step 6: Generating Visualizations and Reports...")

            try:
                if generate_charts:
                    # Generate basic charts
                    _generate_charts(results, output_dir, logger)

                if generate_dashboard:
                    # Generate interactive dashboard
                    _generate_dashboard(results, state_stats, output_dir, logger)

                if generate_report:
                    # Generate detailed report
                    _generate_report(results, state_stats, metadata, output_dir, logger)

            except Exception as e:
                logger.warning(f"⚠️  Visualization/reporting failed: {e}")
                # Don't fail the pipeline for visualization issues

        # Calculate total execution time
        total_time = time.time() - pipeline_start_time

        # Final summary
        logger.info("🎉 Comprehensive HMM Analysis Pipeline Completed!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")

        click.echo("\n" + "=" * 60)
        click.echo("🎉 COMPREHENSIVE HMM ANALYSIS COMPLETED!")
        click.echo("=" * 60)
        click.echo(f"📂 Results saved to: {output_dir}")
        click.echo(f"📊 Data processed: {len(results)} rows")
        click.echo(f"🔢 HMM states: {n_states}")
        click.echo(f"⚙️  Processing engine: {engine}")
        click.echo(f"⏱️  Total time: {total_time:.2f} seconds")
        if PSUTIL_AVAILABLE:
            click.echo(f"🧠 Memory usage: {get_memory_usage():.1%}")

        # Performance summary
        log_performance_metrics(
            pipeline_start_time,
            "complete_pipeline",
            {
                "total_rows": len(results),
                "n_states": n_states,
                "engine": engine,
                "final_memory": get_memory_usage() if PSUTIL_AVAILABLE else 0.0,
            },
        )

    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed: {e}")
        logger.exception("Full traceback:")
        click.echo(f"\n❌ Error: {e}", err=True)
        if logger:
            click.echo("📄 Check the log file for detailed error information", err=True)
        raise click.ClickException(f"Analysis failed: {e}") from e


def _generate_charts(results, output_dir, logger):
    """Generate basic visualization charts."""
    try:
        import matplotlib.pyplot as plt

        # Set style
        plt.style.use("seaborn-v0_8")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("HMM Analysis Results", fontsize=16, fontweight="bold")

        # Plot 1: Price with state colors
        ax1 = axes[0, 0]
        for state in results["hmm_state"].unique():
            state_data = results[results["hmm_state"] == state]
            ax1.scatter(
                state_data.index,
                state_data["close"],
                label=f"State {state}",
                alpha=0.7,
                s=20,
            )
        ax1.set_title("Price Data with HMM States")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: State distribution
        ax2 = axes[0, 1]
        state_counts = results["hmm_state"].value_counts()
        ax2.pie(
            state_counts.values,
            labels=[f"State {s}" for s in state_counts.index],
            autopct="%1.1f%%",
        )
        ax2.set_title("State Distribution")

        # Plot 3: Returns by state
        ax3 = axes[1, 0]
        for state in results["hmm_state"].unique():
            state_returns = results[results["hmm_state"] == state]["returns"]
            ax3.hist(
                state_returns, bins=20, alpha=0.6, label=f"State {state}", density=True
            )
        ax3.set_title("Return Distribution by State")
        ax3.set_xlabel("Returns")
        ax3.set_ylabel("Density")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: State timeline
        ax4 = axes[1, 1]
        ax4.plot(results.index, results["hmm_state"], linewidth=1)
        ax4.set_title("State Timeline")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("State")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / "hmm_analysis_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Charts saved to {chart_path}")

    except Exception as e:
        logger.error(f"❌ Chart generation failed: {e}")
        raise


def _generate_dashboard(results, state_stats, output_dir, logger):
    """Generate interactive dashboard."""
    try:
        # Create dashboard HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HMM Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric h3 {{ margin: 0; color: #333; }}
                .metric p {{ margin: 5px 0; font-size: 1.2em; color: #666; }}
            </style>
        </head>
        <body>
            <h1>HMM Futures Analysis Dashboard</h1>

            <div class="metric">
                <h3>Analysis Summary</h3>
                <p>Dataset Size: {len(results):,} rows</p>
                <p>Number of States: {len(results["hmm_state"].unique())}</p>
                <p>Date Range: {results.index.min()} to {results.index.max()}</p>
            </div>

            <div id="price-chart" style="height: 500px;"></div>
            <div id="state-distribution" style="height: 400px;"></div>

            <script>
                // Price chart with states
                var priceData = [{{
                    x: {list(results.index.astype(str))},
                    y: {list(results["close"])},
                    mode: 'markers',
                    marker: {{
                        color: {list(results["hmm_state"])},
                        colorscale: 'Viridis',
                        size: 4
                    }},
                    type: 'scatter',
                    name: 'Price'
                }}];

                var priceLayout = {{
                    title: 'Price Data with HMM States',
                    xaxis: {{ title: 'Date' }},
                    yaxis: {{ title: 'Price' }}
                }};

                Plotly.newPlot('price-chart', priceData, priceLayout);

                // State distribution pie chart
                var stateData = [{{
                    labels: {["State {i}" for i in state_stats.keys()]},
                    values: {[stats["count"] for stats in state_stats.values()]},
                    type: 'pie'
                }}];

                var stateLayout = {{
                    title: 'State Distribution'
                }};

                Plotly.newPlot('state-distribution', stateData, stateLayout);
            </script>
        </body>
        </html>
        """

        dashboard_path = output_dir / "hmm_dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)

        logger.info(f"✅ Dashboard saved to {dashboard_path}")

    except Exception as e:
        logger.error(f"❌ Dashboard generation failed: {e}")
        raise


def _generate_report(results, state_stats, metadata, output_dir, logger):
    """Generate detailed analysis report."""
    try:
        report_content = f"""
        # HMM Futures Analysis Report

        ## Executive Summary

        This report presents the results of Hidden Markov Model analysis performed on futures data.

        **Analysis Configuration:**
        - Number of States: {metadata["n_components"]}
        - Covariance Type: {metadata.get("covariance_type", "N/A")}
        - Training Iterations: {metadata.get("n_iter", "N/A")}
        - Convergence: {metadata.get("converged", "N/A")}
        - Log-Likelihood: {metadata.get("log_likelihood", "N/A"):.2f}

        **Dataset Overview:**
        - Total Records: {len(results):,}
        - Date Range: {results.index.min()} to {results.index.max()}
        - Features Engineered: {len(results.columns)}

        ## State Analysis

        """

        # Add state analysis
        for state, stats in state_stats.items():
            report_content += f"""
        ### State {state}
        - **Sample Count**: {stats["count"]:,} ({stats["percentage"]:.1f}%)
        - **Mean Return**: {stats["mean_return"]:.4f}
        - **Return Volatility**: {stats["std_return"]:.4f}
        - **Price Volatility**: {stats["volatility"]:.4f}
        - **Sharpe Ratio**: {stats["sharpe_ratio"]:.2f}

        """

        report_content += f"""
        ## Technical Details

        **Model Performance:**
        - Training samples used: {metadata.get("n_samples", "N/A"):,}
        - Model convergence achieved: {"Yes" if metadata.get("converged") else "No"}
        - Final log-likelihood: {metadata.get("log_likelihood", "N/A"):.2f}

        **Data Processing:**
        - Original columns: {len(results.columns) - 1}  # Excluding hmm_state
        - Final features: {len(results.columns)}
        - Missing values handled: None

        ## Recommendations

        Based on the analysis results, consider the following:

        1. **State Interpretation**: Analyze the characteristics of each identified state to understand market regimes
        2. **Strategy Development**: Use state information to develop regime-aware trading strategies
        3. **Risk Management**: Adjust position sizing based on state-specific volatility characteristics
        4. **Further Analysis**: Consider additional features or different numbers of states for refinement

        ---
        *Report generated on {time.strftime("%Y-%m-%d %H:%M:%S")}*
        """

        report_path = output_dir / "hmm_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"✅ Report saved to {report_path}")

    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise


@cli.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to saved HMM model",
)
@click.option(
    "--input-csv",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file with futures data",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for results",
)
@click.pass_context
def infer(ctx, model_path, input_csv, output_dir):
    """
    Load a trained model and infer states on new data.
    """
    logger = ctx.obj["logger"]
    ctx.obj["memory_monitor"]

    try:
        click.echo(f"🔮 Loading model from {model_path}...")
        click.echo(f"📂 Processing new data from {input_csv}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Load model
        model, scaler, config = load_model(str(model_path))
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Number of states: {config.get('n_states', 'N/A')}")

        # Process new data
        processing_config = ProcessingConfig()
        factory = ProcessingEngineFactory()
        process_func = factory.get_engine("streaming")
        data = process_func(str(input_csv), processing_config)

        # Feature engineering
        features = add_features(data)
        features_clean = features.dropna()

        # Prepare features for inference
        if scaler:
            X = scaler.transform(features_clean["close"].values.reshape(-1, 1))
        else:
            X = features_clean["close"].values.reshape(-1, 1)

        # Validate features
        validate_features_for_hmm(X)

        # Infer states
        states = inference_engine.predict_states(model, scaler, X)

        # Create results
        results = features_clean.copy()
        results["hmm_state"] = states

        # Save results
        states_path = output_dir / "inferred_states.csv"
        results.to_csv(states_path)

        click.echo("✅ State inference completed")
        click.echo(f"📊 {len(results)} rows processed")
        click.echo(f"🔢 {len(np.unique(states))} unique states found")
        click.echo(f"📄 Results saved to {states_path}")

        log_performance_metrics(
            start_time,
            "state_inference",
            {"rows_processed": len(results), "unique_states": len(np.unique(states))},
        )

    except Exception as e:
        logger.error(f"❌ State inference failed: {e}")
        raise click.ClickException(f"State inference failed: {e}") from e


@cli.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to saved HMM model",
)
@click.pass_context
def model_info(ctx, model_path):
    """
    Display information about a saved HMM model.
    """
    logger = ctx.obj["logger"]

    try:
        click.echo(f"📋 Loading model information from {model_path}...")

        # Load model
        model, scaler, config = load_model(str(model_path))

        # Display model information
        click.echo("\n" + "=" * 50)
        click.echo("HMM MODEL INFORMATION")
        click.echo("=" * 50)

        click.echo("\n🔧 Model Configuration:")
        for key, value in config.items():
            click.echo(f"  {key}: {value}")

        click.echo("\n📊 Model Parameters:")
        click.echo(f"  Number of states (components): {model.n_components}")
        click.echo(f"  Covariance type: {model.covariance_type}")
        click.echo(f"  Number of iterations: {model.n_iter}")
        click.echo(f"  Tolerance: {model.tol}")

        if hasattr(model, "means_"):
            click.echo(f"  State means shape: {model.means_.shape}")

        if hasattr(model, "covars_"):
            click.echo(f"  Covariance matrices shape: {model.covars_.shape}")

        click.echo("\n" + "=" * 50)

        logger.info(f"✅ Model information displayed for {model_path}")

    except Exception as e:
        logger.error(f"❌ Failed to load model information: {e}")
        raise click.ClickException(f"Failed to load model information: {e}") from e


@cli.command()
def version():
    """Show version and system information."""
    click.echo("HMM Futures Analysis CLI v1.0.0")
    click.echo("=" * 40)
    click.echo(f"Python version: {sys.version}")
    click.echo(f"Platform: {sys.platform}")

    try:
        import hmmlearn
        import numpy as np
        import pandas as pd
        import sklearn

        click.echo("\n📦 Key Dependencies:")
        click.echo(f"  pandas: {pd.__version__}")
        click.echo(f"  numpy: {np.__version__}")
        click.echo(f"  scikit-learn: {sklearn.__version__}")
        click.echo(f"  hmmlearn: {hmmlearn.__version__}")
    except ImportError as e:
        click.echo(f"⚠️  Missing dependency: {e}")

    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            click.echo("\n💾 System Memory:")
            click.echo(f"  Total: {memory.total / (1024**3):.1f} GB")
            click.echo(f"  Available: {memory.available / (1024**3):.1f} GB")
            click.echo(
                f"  Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)"
            )
        except Exception:
            click.echo("\n💾 Memory information unavailable")
    else:
        click.echo("\n💾 psutil not available for memory information")


if __name__ == "__main__":
    cli()
