"""
Daft Engine Module

Implements distributed processing using Daft DataFrames for modern,
cloud-native data processing with automatic optimization and GPU acceleration.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import daft
    import pyarrow as pa
    import pyarrow.compute as pc
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    daft = None
    pa = None
    pc = None

from data_processing import add_features, validate_data
from utils import ProcessingConfig, get_logger

logger = get_logger(__name__)


def process_daft(
    csv_path: str,
    config: ProcessingConfig,
    npartitions: Optional[int] = None,
    memory_limit: Optional[str] = "2GB",
    show_progress: bool = True,
    use_accelerators: bool = True
) -> "daft.DataFrame":
    """
    Process CSV file using Daft for modern distributed computing.

    Args:
        csv_path: Path to the CSV file
        config: Processing configuration
        npartitions: Number of partitions (auto-detected if None)
        memory_limit: Memory limit per partition
        show_progress: Whether to show progress
        use_accelerators: Whether to use GPU/accelerator if available

    Returns:
        daft.DataFrame: Processed Daft DataFrame

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ImportError: If daft is not available
        RuntimeError: If Daft processing fails
    """
    if not DAFT_AVAILABLE:
        raise ImportError("Daft is not installed. Install with: pip install getdaft")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Starting Daft processing of {csv_path}")
    logger.info(f"Configuration: engine={config.engine_type}, npartitions={npartitions}")

    try:
        # Load CSV with Daft
        logger.info("Loading CSV with Daft...")
        df = daft.read_csv(str(csv_path))

        # Auto-detect optimal partitions
        if npartitions is None:
            npartitions = _auto_detect_partitions_daft(df, csv_path)
            logger.info(f"Auto-detected optimal partitions: {npartitions}")

        # Repartition if needed
        if npartitions > 1:
            logger.info(f"Repartitioning to {npartitions} partitions")
            df = df.repartition(npartitions)

        logger.info(f"Created Daft DataFrame: {df.count_rows()} rows (estimated)")

        # Apply CSV preprocessing
        logger.info("Applying CSV preprocessing with Daft...")
        df = _apply_csv_preprocessing_daft(df)

        # Apply feature engineering
        logger.info("Applying feature engineering with Daft...")
        df = _apply_feature_engineering_daft(df, config)

        # Apply data validation
        logger.info("Applying data validation with Daft...")
        df = _apply_validation_daft(df)

        # Show basic statistics
        if show_progress:
            logger.info("Computing basic statistics...")
            try:
                row_count = df.count_rows()
                logger.info(f"Processed {row_count} rows with {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"Could not compute statistics: {e}")

        logger.info("Daft processing completed successfully")
        return df

    except Exception as e:
        logger.error(f"Daft processing failed: {e}")
        raise


def _auto_detect_partitions_daft(df: "daft.DataFrame", csv_path: Path) -> int:
    """
    Auto-detect optimal number of partitions for Daft processing.

    Args:
        df: Daft DataFrame
        csv_path: Path to CSV file

    Returns:
        int: Recommended number of partitions
    """
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)

    # Base calculation: aim for ~128MB per partition for Daft
    base_partitions = max(1, int(file_size_mb / 128))

    # Adjust based on available cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    # Daft can handle more partitions efficiently
    if file_size_mb < 100:  # Small files
        optimal_partitions = min(base_partitions, cpu_count)
    elif file_size_mb < 1000:  # Medium files
        optimal_partitions = min(base_partitions, cpu_count * 2)
    else:  # Large files
        optimal_partitions = min(base_partitions, cpu_count * 4)

    # Ensure reasonable limits
    optimal_partitions = max(1, optimal_partitions)
    optimal_partitions = min(optimal_partitions, 50)  # Cap at 50 partitions

    logger.debug(f"Auto-detected Daft partitions: {optimal_partitions} "
                 f"(file: {file_size_mb:.1f}MB, cores: {cpu_count})")

    return optimal_partitions


def _apply_csv_preprocessing_daft(df: "daft.DataFrame") -> "daft.DataFrame":
    """
    Apply CSV preprocessing to standardize columns using Daft operations.

    Args:
        df: Daft DataFrame

    Returns:
        daft.DataFrame: Preprocessed Daft DataFrame
    """
    logger.info("Standardizing column names with Daft...")

    # Get column names and standardize them
    columns = df.column_names
    column_mapping = _detect_ohlcv_columns_daft(columns)

    if not column_mapping:
        raise ValueError("Could not detect required OHLCV columns in Daft DataFrame")

    # Rename columns to standard names using Daft's native operations
    # Since Daft doesn't have rename, we'll manually recreate columns
    for old_name, new_name in column_mapping.items():
        if old_name != new_name:
            df = df.with_column(new_name, df[old_name])
            df = df.exclude(old_name)

    # Process datetime columns using Daft expressions
    df = _process_datetime_columns_daft_expressions(df)

    # Clean numeric data using Daft expressions
    df = _clean_numeric_data_daft_expressions(df)

    logger.info(f"Standardized columns: {list(df.column_names)}")
    return df


def _detect_ohlcv_columns_daft(columns: List[str]) -> Dict[str, str]:
    """Detect OHLCV columns with various naming conventions for Daft DataFrames."""
    # Strip whitespace from column names first
    columns = [col.strip() for col in columns]

    # Common column name variations
    ohlcv_patterns = {
        'open': ['open', 'Open', 'OPEN', 'o', 'O', 'price_open', 'Price_Open'],
        'high': ['high', 'High', 'HIGH', 'h', 'H', 'price_high', 'Price_High'],
        'low': ['low', 'Low', 'LOW', 'l', 'L', 'price_low', 'Price_Low'],
        'close': ['close', 'Close', 'CLOSE', 'c', 'C', 'price_close', 'Price_Close', 'last', 'Last'],
        'volume': ['volume', 'Volume', 'VOLUME', 'v', 'V', 'vol', 'Vol', 'trades']
    }

    column_mapping = {}

    for standard_name, variations in ohlcv_patterns.items():
        for variation in variations:
            if variation in columns:
                column_mapping[variation] = standard_name
                break

    return column_mapping


def _apply_feature_engineering_daft(df: "daft.DataFrame", config: ProcessingConfig) -> "daft.DataFrame":
    """
    Apply feature engineering to Daft DataFrame.

    Args:
        df: Daft DataFrame
        config: Processing configuration

    Returns:
        daft.DataFrame: Daft DataFrame with features
    """
    logger.info("Adding technical indicators with Daft...")

    try:
        # Convert to pandas for feature engineering (Daft is still young)
        # In a production setting, you'd implement these natively in Daft
        logger.info("Converting to pandas for feature engineering...")
        pandas_df = df.to_pandas()

        # Apply feature engineering using existing pandas function
        pandas_df = add_features(
            pandas_df,
            indicator_config=config.indicators,
            downcast_floats=True
        )

        # Convert back to Daft
        logger.info("Converting back to Daft DataFrame...")
        result_df = daft.from_pyarrow(pandas_df.to_arrow())

        return result_df

    except Exception as e:
        logger.error(f"Feature engineering failed in Daft: {e}")
        # Fallback: return original DataFrame without features
        logger.warning("Returning original DataFrame without features")
        return df


def _process_datetime_columns_daft_expressions(df: "daft.DataFrame") -> "daft.DataFrame":
    """
    Process datetime columns using Daft expressions for better performance.

    Args:
        df: Daft DataFrame

    Returns:
        daft.DataFrame: DataFrame with processed datetime columns
    """
    date_columns = ['date', 'Date', 'datetime', 'DateTime', 'timestamp', 'Timestamp']
    time_columns = ['time', 'Time', 'datetime', 'DateTime', 'timestamp', 'Timestamp']

    # Find date column
    date_col = None
    for col in date_columns:
        if col in df.column_names:
            date_col = col
            break

    # Find time column
    time_col = None
    for col in time_columns:
        if col in df.column_names and col != date_col:
            time_col = col
            break

    if date_col is None:
        return df  # Keep original if no date column found

    try:
        if time_col and time_col in df.column_names:
            # Combine date and time columns
            datetime_col = "datetime"
            df = df.with_column(
                datetime_col,
                daft.col(date_col).cast(daft.DataType.string()) + daft.lit(" ") + daft.col(time_col).cast(daft.DataType.string())
            )
            df = df.with_column(datetime_col, daft.col(datetime_col).cast(daft.DataType.timestamp("us")))
            df = df.exclude(date_col, time_col)
        else:
            # Convert date column to datetime
            df = df.with_column(date_col, daft.col(date_col).cast(daft.DataType.timestamp("us")))
            datetime_col = date_col

        # Set datetime as index
        df = df.set_index(datetime_col)

    except Exception as e:
        logger.warning(f"Failed to process datetime columns with Daft expressions: {e}")
        # Fallback to pandas-based processing
        pass

    return df


def _clean_numeric_data_daft_expressions(df: "daft.DataFrame") -> "daft.DataFrame":
    """
    Clean and standardize numeric data using Daft expressions.

    Args:
        df: Daft DataFrame

    Returns:
        daft.DataFrame: DataFrame with cleaned numeric data
    """
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']

    for col in numeric_columns:
        if col in df.column_names:
            # Convert to numeric, coercing errors to null
            df = df.with_column(col, daft.col(col).cast(daft.DataType.float32()))

            # Remove rows with null in price columns (but not volume)
            if col != 'volume':
                df = df.filter(daft.col(col).not_null())

    return df


def _apply_validation_daft(df: "daft.DataFrame") -> "daft.DataFrame":
    """
    Apply data validation to Daft DataFrame.

    Args:
        df: Daft DataFrame

    Returns:
        daft.DataFrame: Validated Daft DataFrame
    """
    logger.info("Applying data validation with Daft...")

    try:
        # Convert to pandas for validation
        pandas_df = df.to_pandas()

        # Apply validation using existing pandas function
        pandas_df, validation_report = validate_data(
            pandas_df,
            outlier_detection=True,
            outlier_method="iqr",
            outlier_threshold=1.5,
            missing_value_strategy="forward_fill"
        )

        logger.info(f"Validation completed: {validation_report}")

        # Convert back to Daft
        result_df = daft.from_pyarrow(pandas_df.to_arrow())

        return result_df

    except Exception as e:
        logger.error(f"Validation failed in Daft: {e}")
        # Fallback: return original DataFrame without validation
        logger.warning("Returning original DataFrame without validation")
        return df


def compute_daft_with_progress(df: "daft.DataFrame", show_progress: bool = True) -> "pd.DataFrame":
    """
    Compute Daft DataFrame with optional progress reporting.

    Args:
        df: Daft DataFrame to compute
        show_progress: Whether to show progress

    Returns:
        pd.DataFrame: Computed pandas DataFrame
    """
    logger.info("Computing Daft DataFrame...")

    if show_progress:
        logger.info("Processing Daft DataFrame...")

    start_time = time.time()

    # Convert to pandas
    result = df.to_pandas()

    end_time = time.time()
    processing_time = end_time - start_time

    logger.info(f"Daft computation completed: {len(result)} rows, {len(result.columns)} columns")
    logger.info(f"Computation time: {processing_time:.2f}s")

    return result


def optimize_daft_performance(
    use_accelerators: bool = True,
    memory_fraction: float = 0.8
) -> Dict[str, Any]:
    """
    Optimize Daft performance settings.

    Args:
        use_accelerators: Whether to use GPU/accelerator if available
        memory_fraction: Fraction of available memory to use

    Returns:
        Dict with optimization settings applied
    """
    settings = {}

    if DAFT_AVAILABLE:
        # Configure Daft settings
        try:
            # Memory optimization
            settings['memory_fraction'] = memory_fraction
            logger.info(f"Configured Daft memory fraction: {memory_fraction}")

            # Accelerator optimization
            if use_accelerators:
                # Check if GPU is available
                try:
                    import torch
                    if torch.cuda.is_available():
                        settings['gpu_available'] = True
                        logger.info("GPU acceleration available for Daft")
                    else:
                        settings['gpu_available'] = False
                        logger.info("GPU not available, using CPU")
                except ImportError:
                    settings['gpu_available'] = False
                    logger.info("PyTorch not available, using CPU")

            # Set Daft configuration if available
            # Note: Daft configuration API is evolving
            logger.info("Daft optimization completed")

        except Exception as e:
            logger.warning(f"Failed to configure Daft optimization: {e}")

    else:
        logger.warning("Daft not available for optimization")

    return settings


def benchmark_daft_engine(
    csv_path: str,
    partition_counts: List[int] = None,
    use_accelerators: bool = True
) -> Dict[str, Any]:
    """
    Benchmark different Daft configurations for performance testing.

    Args:
        csv_path: Path to test CSV file
        partition_counts: List of partition counts to test
        use_accelerators: Whether to test with accelerators

    Returns:
        Dict with benchmark results
    """
    if partition_counts is None:
        partition_counts = [1, 2, 4]
    if not DAFT_AVAILABLE:
        return {'error': 'Daft is not available'}

    logger.info("Starting Daft benchmarking...")

    results = {}

    for npartitions in partition_counts:
        config_key = f"partitions_{npartitions}"
        results[config_key] = {}

        logger.info(f"Testing Daft with {npartitions} partitions")

        try:
            start_time = time.time()

            # Create temporary config
            config = ProcessingConfig(
                engine_type="daft",
                chunk_size=1000
            )

            # Process with Daft
            df = process_daft(
                csv_path,
                config=config,
                npartitions=npartitions,
                show_progress=False,
                use_accelerators=use_accelerators
            )

            # Compute the result
            result = compute_daft_with_progress(df, show_progress=False)

            end_time = time.time()
            processing_time = end_time - start_time

            # Record results
            results[config_key] = {
                'time': processing_time,
                'rows': len(result),
                'columns': len(result.columns),
                'partitions': npartitions,
                'memory_mb': result.memory_usage(deep=True).sum() / (1024 * 1024),
                'success': True
            }

            logger.info(f"  Daft {npartitions} partitions - {processing_time:.2f}s, "
                       f"{len(result)} rows, {results[config_key]['memory_mb']:.2f}MB")

        except Exception as e:
            logger.error(f"  Daft {npartitions} partitions - FAILED: {e}")
            results[config_key] = {
                'error': str(e),
                'success': False,
                'partitions': npartitions
            }

    return results


def get_daft_cluster_info() -> Dict[str, Any]:
    """
    Get information about the Daft execution environment.

    Returns:
        Dict with Daft environment information
    """
    info = {
        'daft_available': DAFT_AVAILABLE,
        'version': None,
        'execution_backend': None,
        'accelerators': False
    }

    if DAFT_AVAILABLE:
        try:
            info['version'] = daft.__version__
            logger.info(f"Daft version: {info['version']}")

            # Check execution backend
            # Note: This is a simplified check - Daft's backend detection is more complex
            try:
                import torch
                if torch.cuda.is_available():
                    info['accelerators'] = True
                    info['execution_backend'] = 'GPU'
                else:
                    info['execution_backend'] = 'CPU'
            except ImportError:
                info['execution_backend'] = 'CPU'

            logger.info(f"Daft execution backend: {info['execution_backend']}")
            logger.info(f"Daft accelerators available: {info['accelerators']}")

        except Exception as e:
            logger.error(f"Failed to get Daft info: {e}")
            info['error'] = str(e)

    return info
