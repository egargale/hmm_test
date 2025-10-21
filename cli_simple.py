"""
Simple HMM Futures Analysis CLI

A simplified command-line interface for HMM analysis focusing on core functionality.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional
import time

import click
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import get_logger, setup_logging
from data_processing.csv_parser import process_csv
from data_processing.data_validation import validate_data

# Global logger
logger = None


@click.group()
@click.version_option(version="1.0.0", prog_name="hmm-analysis")
@click.option('--log-level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              default='INFO',
              help='Set logging level (default: INFO)')
@click.pass_context
def cli(ctx, log_level):
    """
    HMM Futures Analysis CLI (Simple Version)

    A simplified command-line tool for HMM futures market analysis.
    """
    # Set up logging
    setup_logging(level=log_level.upper())
    global logger
    logger = get_logger(__name__)

    logger.info("HMM Futures Analysis CLI (Simple) started")

    # Store global config in context
    ctx.ensure_object(dict)
    ctx.obj['log_level'] = log_level
    ctx.obj['logger'] = logger


@cli.command()
@click.option('--input-csv', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input CSV file with futures data')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=Path('./output'),
              help='Output directory for results')
@click.pass_context
def validate(ctx, input_csv, output_dir):
    """
    Validate input data format and structure.
    """
    logger = ctx.obj['logger']

    try:
        click.echo(f"Validating {input_csv}...")

        # Load and validate data
        data = process_csv(str(input_csv))
        data_clean, validation_result = validate_data(data)

        # Check if validation succeeded (no critical issues)
        critical_issues = [issue for issue in validation_result['issues_found']
                          if issue.get('severity') == 'critical']

        if not critical_issues:
            click.echo("✅ Data validation passed!")
            click.echo(f"📊 {len(data)} rows of data")
            click.echo(f"📅 Date range: {data_clean.index.min()} to {data_clean.index.max()}")
            click.echo(f"📈 Columns: {list(data_clean.columns)}")

            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save validation report
            report_path = output_dir / "validation_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Data Validation Report\n")
                f.write(f"======================\n\n")
                f.write(f"File: {input_csv}\n")
                f.write(f"Rows: {len(data_clean)}\n")
                f.write(f"Columns: {list(data_clean.columns)}\n")
                f.write(f"Date range: {data_clean.index.min()} to {data_clean.index.max()}\n")
                f.write(f"Quality score: {validation_result.get('quality_score', 'N/A')}\n")
                f.write(f"Issues found: {len(validation_result['issues_found'])}\n")
                f.write(f"Critical issues: {len(critical_issues)}\n")
                f.write(f"\nValidation status: PASSED\n")

            click.echo(f"📄 Validation report saved to: {report_path}")
        else:
            click.echo("❌ Data validation failed:", err=True)
            for issue in critical_issues:
                click.echo(f"  • {issue.get('description', 'Unknown error')}", err=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input-csv', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input CSV file with futures data')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=Path('./output'),
              help='Output directory for results')
@click.option('--n-states', '-n',
              type=click.IntRange(min=2, max=5),
              default=3,
              help='Number of HMM states (default: 3)')
@click.option('--test-size',
              type=click.FloatRange(min=0.1, max=0.5),
              default=0.2,
              help='Proportion of data for testing (default: 0.2)')
@click.option('--random-seed',
              type=int,
              default=42,
              help='Random seed for reproducibility (default: 42)')
@click.pass_context
def analyze(ctx, input_csv, output_dir, n_states, test_size, random_seed):
    """
    Run simplified HMM analysis pipeline.

    This command executes a simplified analysis pipeline:
    1. Data loading and validation
    2. Basic feature engineering
    3. Simple HMM training
    4. Basic state inference
    """
    logger = ctx.obj['logger']

    try:
        # Validate inputs
        logger.info("🚀 Starting Simple HMM Analysis Pipeline")
        click.echo(f"📂 Input file: {input_csv}")
        click.echo(f"📊 Output directory: {output_dir}")
        click.echo(f"🔢 Number of states: {n_states}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Start timing
        start_time = time.time()

        # Step 1: Load and validate data
        logger.info("📁 Step 1: Loading and validating data...")
        click.echo("Loading data...")

        try:
            data = process_csv(str(input_csv))
            data_clean, validation_result = validate_data(data)

            # Check for critical validation issues
            critical_issues = [issue for issue in validation_result['issues_found']
                              if issue.get('severity') == 'critical']

            if critical_issues:
                error_descriptions = [issue.get('description', 'Unknown error') for issue in critical_issues]
                raise ValueError(f"Data validation failed: {error_descriptions}")

            logger.info(f"✅ Loaded {len(data_clean)} rows of data")

        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise click.ClickException(f"Failed to load data: {e}")

        # Step 2: Basic feature engineering
        logger.info("⚙️  Step 2: Basic feature engineering...")
        click.echo("Engineering basic features...")

        try:
            # Use cleaned data from validation
            data = data_clean.copy()

            # Simple returns
            data['returns'] = data['close'].pct_change()

            # Simple moving averages
            for period in [5, 10, 20]:
                data[f'sma_{period}'] = data['close'].rolling(window=period).mean()

            # Simple volatility
            data['volatility_14'] = data['returns'].rolling(window=14).std()

            # Drop NaN values
            data = data.dropna()

            click.echo(f"✅ Added {len(data.columns) - 5} basic features")

        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            raise click.ClickException(f"Feature engineering failed: {e}")

        # Step 3: Simple HMM training
        logger.info("🧠 Step 3: Simple HMM model training...")
        click.echo(f"Training HMM with {n_states} states...")

        try:
            from hmmlearn import hmm

            # Prepare training data
            train_size = int(len(data) * (1 - test_size))
            train_data = data['close'].iloc[:train_size].values.reshape(-1, 1)

            # Simple HMM model
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=50,
                random_state=random_seed
            )

            with tqdm(total=50, desc="Training model") as pbar:
                # Simple training with progress indication
                for i in range(50):
                    model.fit(train_data)
                    pbar.update(1)

            logger.info(f"✅ HMM training completed. Score: {model.score(train_data):.2f}")

        except Exception as e:
            logger.error(f"❌ HMM training failed: {e}")
            raise click.ClickException(f"HMM training failed: {e}")

        # Step 4: Simple state inference
        logger.info("🔍 Step 4: Simple state inference...")
        click.echo("Inferring hidden states...")

        try:
            # Infer states for full dataset
            full_data = data['close'].values.reshape(-1, 1)
            states = model.predict(full_data)

            logger.info(f"✅ State inference completed. Found {len(np.unique(states))} unique states")

            # Create basic state statistics
            state_stats = {}
            for state in np.unique(states):
                state_mask = states == state
                state_stats[state] = {
                    'count': np.sum(state_mask),
                    'mean_return': data['returns'].iloc[state_mask].mean(),
                    'std_return': data['returns'].iloc[state_mask].std(),
                    'count': np.sum(state_mask)
                }

        except Exception as e:
            logger.error(f"❌ State inference failed: {e}")
            raise click.ClickException(f"State inference failed: {e}")

        # Step 5: Save results
        logger.info("💾 Step 5: Saving results...")
        click.echo("Saving analysis results...")

        try:
            # Save states
            states_df = data.copy()
            states_df['hmm_state'] = states
            states_path = output_dir / "states.csv"
            states_df.to_csv(states_path)

            # Save model info
            model_info = {
                'n_states': n_states,
                'n_components': model.n_components,
                'converged': model.monitor_.converged,
                'n_iter': model.n_iter,
                'score': model.score(train_data),
                'training_samples': len(train_data),
                'total_samples': len(full_data)
            }

            model_path = output_dir / "model_info.txt"
            with open(model_path, 'w') as f:
                f.write("HMM Model Information\n")
                f.write("=====================\n\n")
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")

            # Save state statistics
            stats_path = output_dir / "state_statistics.txt"
            with open(stats_path, 'w') as f:
                f.write("State Statistics\n")
                f.write("================\n\n")
                for state, stats in state_stats.items():
                    f.write(f"State {state}:\n")
                    for key, value in stats.items():
                        f.write(f"  {key}: {value:.4f}\n")
                    f.write("\n")

            click.echo(f"✅ Results saved to {output_dir}")
            click.echo(f"  📊 States: {states_path}")
            click.echo(f"  🧠 Model info: {model_path}")
            click.echo(f"  📈 Statistics: {stats_path}")

        except Exception as e:
            logger.error(f"❌ Saving results failed: {e}")
            # Don't fail the entire pipeline for saving issues

        # Calculate total execution time
        total_time = time.time() - start_time

        # Final summary
        logger.info("🎉 Simple HMM Analysis Pipeline Completed!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")

        click.echo("\n" + "="*60)
        click.echo("🎉 SIMPLE HMM ANALYSIS COMPLETED!")
        click.echo("="*60)
        click.echo(f"📂 Results saved to: {output_dir}")
        click.echo(f"📊 Data processed: {len(data)} rows")
        click.echo(f"🔢 HMM states: {n_states}")
        click.echo(f"⏱️  Total time: {total_time:.2f} seconds")

    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed: {e}")
        click.echo(f"\n❌ Error: {e}", err=True)
        raise click.ClickException(f"Analysis failed: {e}")


@cli.command()
def version():
    """Show version information."""
    click.echo("HMM Futures Analysis CLI v1.0.0 (Simple)")
    click.echo("© 2024 - Advanced Regime Detection System")


if __name__ == '__main__':
    cli()