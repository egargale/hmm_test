"""
Dask Engine Module

Implements distributed processing using Dask DataFrames for lazy, scalable
processing of large datasets with automatic parallelization and out-of-core capabilities.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import dask
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from data_processing import add_features, validate_data
from utils import ProcessingConfig, get_logger

logger = get_logger(__name__)


def process_dask(
    csv_path: str,
    config: ProcessingConfig,
    scheduler: Optional[str] = "threads",
    npartitions: Optional[int] = None,
    memory_limit: Optional[str] = "2GB",
    show_progress: bool = True,
) -> dd.DataFrame:
    """
    Process CSV file using Dask for distributed computing.

    Args:
        csv_path: Path to the CSV file
        config: Processing configuration
        scheduler: Dask scheduler type ('threads', 'processes', 'synchronous', or custom)
        npartitions: Number of partitions (auto-detected if None)
        memory_limit: Memory limit per worker
        show_progress: Whether to show progress bar

    Returns:
        dask.dataframe.DataFrame: Lazy Dask DataFrame (call .compute() to get results)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ImportError: If dask is not available
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Starting Dask processing of {csv_path}")
    logger.info(
        f"Configuration: engine={config.engine_type}, scheduler={scheduler}, npartitions={npartitions}"
    )

    # Setup Dask client
    client = None
    if scheduler not in ["threads", "synchronous"]:
        try:
            client = Client(scheduler=scheduler)
            logger.info(f"Created Dask client with scheduler: {scheduler}")
        except Exception as e:
            logger.warning(
                f"Failed to create Dask client with scheduler '{scheduler}': {e}"
            )
            logger.info("Falling back to threaded scheduler")
            scheduler = "threads"

    try:
        # Load CSV with Dask
        logger.info("Loading CSV with Dask...")
        ddf = dd.read_csv(
            csv_path,
            blocksize="64MB",  # Default block size
            dtype={
                "open": "float32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "volume": "float32",
            },
        )

        # Auto-detect partitions if not specified
        if npartitions is None:
            npartitions = _auto_detect_partitions(ddf, csv_path)
            logger.info(f"Auto-detected optimal partitions: {npartitions}")

        # Repartition if needed
        if npartitions != ddf.npartitions:
            logger.info(
                f"Repartitioning from {ddf.npartitions} to {npartitions} partitions"
            )
            ddf = ddf.repartition(npartitions=npartitions)

        logger.info(f"Created Dask DataFrame: {ddf.npartitions} partitions")
        logger.info(f"Columns: {len(ddf.columns)}")

        # Apply CSV preprocessing and feature engineering
        logger.info("Applying CSV preprocessing with Dask...")
        ddf = _apply_csv_preprocessing_dask(ddf)

        # Update metadata after preprocessing
        ddf = _update_dask_metadata(ddf)

        logger.info("Applying feature engineering with Dask...")
        ddf = _apply_feature_engineering_dask(ddf, config)

        # Apply data validation
        logger.info("Applying data validation with Dask...")
        ddf = _apply_validation_dask(ddf)

        logger.info("Dask processing completed successfully")
        logger.info(
            f"Final DataFrame: {ddf.npartitions} partitions, {len(ddf.columns)} columns"
        )

        return ddf

    except Exception as e:
        logger.error(f"Dask processing failed: {e}")
        raise

    finally:
        # Close client if created
        if client:
            client.close()
            logger.info("Closed Dask client")


def _auto_detect_partitions(ddf: dd.DataFrame, csv_path: Path) -> int:
    """
    Auto-detect optimal number of partitions based on file size and system resources.

    Args:
        ddf: Dask DataFrame
        csv_path: Path to CSV file

    Returns:
        int: Recommended number of partitions
    """
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)

    # Base calculation: aim for ~64MB per partition (Dask default block size)
    base_partitions = max(1, int(file_size_mb / 64))

    # Adjust based on available cores
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # Don't create more partitions than CPU cores (unless file is very large)
    if file_size_mb < 1000:  # < 1GB
        optimal_partitions = min(base_partitions, cpu_count * 2)
    elif file_size_mb < 10000:  # < 10GB
        optimal_partitions = min(base_partitions, cpu_count * 4)
    else:  # Very large files
        optimal_partitions = min(base_partitions, cpu_count * 8)

    # Ensure minimum and maximum partitions
    optimal_partitions = max(1, optimal_partitions)
    optimal_partitions = min(optimal_partitions, 100)  # Cap at 100 partitions

    logger.debug(
        f"Auto-detected partitions: {optimal_partitions} "
        f"(file: {file_size_mb:.1f}MB, cores: {cpu_count})"
    )

    return optimal_partitions


def _apply_feature_engineering_dask(
    ddf: dd.DataFrame, config: ProcessingConfig
) -> dd.DataFrame:
    """
    Apply feature engineering to Dask DataFrame using map_partitions.

    Args:
        ddf: Dask DataFrame
        config: Processing configuration

    Returns:
        dd.DataFrame: Dask DataFrame with features
    """

    def process_partition(df):
        """Process a single partition."""
        # Apply feature engineering
        df = add_features(df, indicator_config=config.indicators, downcast_floats=True)
        return df

    # Clear divisions to ensure metadata doesn't interfere
    ddf = ddf.clear_divisions()

    # Apply feature engineering to each partition without specifying meta
    # This allows Dask to infer the metadata from the computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress Dask warnings
        ddf = ddf.map_partitions(process_partition)

    return ddf


def _create_sample_meta_with_features() -> pd.DataFrame:
    """Create a sample DataFrame with expected feature columns for metadata."""
    # Basic OHLCV columns
    meta = pd.DataFrame(
        {
            "open": pd.Series([], dtype="float32"),
            "high": pd.Series([], dtype="float32"),
            "low": pd.Series([], dtype="float32"),
            "close": pd.Series([], dtype="float32"),
            "volume": pd.Series([], dtype="float32"),
        }
    )

    # Add common feature columns that might be created
    common_features = [
        "log_ret",
        "volatility",
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "sma_10",
        "sma_20",
        "ema_12",
        "atr",
        "obv",
        "vwap",
    ]

    for feature in common_features:
        meta[feature] = pd.Series([], dtype="float32")

    return meta


def _apply_csv_preprocessing_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Apply CSV preprocessing to standardize columns and data types using map_partitions.

    Args:
        ddf: Dask DataFrame

    Returns:
        dd.DataFrame: Preprocessed Dask DataFrame with standard OHLCV columns
    """

    def preprocess_partition(df):
        """Preprocess a single partition."""
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Detect and standardize OHLCV columns
        column_mapping = _detect_ohlcv_columns_dask(df)

        if not column_mapping:
            raise ValueError("Could not detect required OHLCV columns in partition")

        # Rename columns to standard names
        df = df.rename(columns=column_mapping)

        # Process datetime columns
        df = _process_datetime_columns_dask(df)

        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

        # Clean numeric data
        df = _clean_numeric_data_dask(df)

        return df

    # Clear divisions to ensure metadata doesn't interfere
    ddf = ddf.clear_divisions()

    # Apply preprocessing to each partition without specifying meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress Dask warnings
        ddf = ddf.map_partitions(preprocess_partition)

    return ddf


def _create_sample_meta_for_preprocessing() -> pd.DataFrame:
    """Create a sample DataFrame with expected OHLCV columns for metadata."""
    return pd.DataFrame(
        {
            "open": pd.Series([], dtype="float32"),
            "high": pd.Series([], dtype="float32"),
            "low": pd.Series([], dtype="float32"),
            "close": pd.Series([], dtype="float32"),
            "volume": pd.Series([], dtype="float32"),
        }
    )


def _update_dask_metadata(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Update Dask DataFrame metadata after preprocessing to handle column changes.

    Args:
        ddf: Dask DataFrame with outdated metadata

    Returns:
        dd.DataFrame: Updated Dask DataFrame with correct metadata
    """
    try:
        # Compute a small sample to get the actual structure
        sample = ddf.head(1)

        # Clear divisions to reset metadata
        ddf = ddf.clear_divisions()

        # Update metadata based on actual computed sample
        if hasattr(ddf, "_meta_assign"):
            # For newer Dask versions that support _meta_assign
            ddf._meta_assign(sample)
        elif hasattr(ddf, "_meta"):
            # Try to update directly if possible
            try:
                ddf._meta = sample
            except (AttributeError, TypeError):
                # If _meta is read-only, just clear divisions
                pass

        logger.debug(f"Updated Dask metadata: {len(sample.columns)} columns")
        return ddf

    except Exception as e:
        logger.warning(f"Could not update Dask metadata: {e}")
        # Clear divisions as fallback
        return ddf.clear_divisions()


def _detect_ohlcv_columns_dask(df: pd.DataFrame) -> Dict[str, str]:
    """Detect OHLCV columns with various naming conventions for Dask partitions."""
    # Common column name variations
    ohlcv_patterns = {
        "open": ["open", "Open", "OPEN", "o", "O", "price_open", "Price_Open"],
        "high": ["high", "High", "HIGH", "h", "H", "price_high", "Price_High"],
        "low": ["low", "Low", "LOW", "l", "L", "price_low", "Price_Low"],
        "close": [
            "close",
            "Close",
            "CLOSE",
            "c",
            "C",
            "price_close",
            "Price_Close",
            "last",
            "Last",
        ],
        "volume": [
            "volume",
            "Volume",
            "VOLUME",
            "v",
            "V",
            "vol",
            "Vol",
            "Volume",
            "trades",
        ],
    }

    column_mapping = {}
    available_columns = list(df.columns)

    for standard_name, variations in ohlcv_patterns.items():
        for variation in variations:
            if variation in available_columns:
                column_mapping[variation] = standard_name
                break

    return column_mapping


def _process_datetime_columns_dask(df: pd.DataFrame) -> pd.DataFrame:
    """Process and combine date/time columns into datetime index for Dask partitions."""
    date_columns = ["date", "Date", "datetime", "DateTime", "timestamp", "Timestamp"]
    time_columns = ["time", "Time", "datetime", "DateTime", "timestamp", "Timestamp"]

    date_col = None
    time_col = None

    # Find date column
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break

    # Find time column
    for col in time_columns:
        if col in df.columns and col != date_col:
            time_col = col
            break

    if date_col is None:
        return df  # Keep original index if no date column found

    try:
        if time_col and time_col in df.columns:
            # Combine date and time columns
            datetime_col = f"{date_col}_{time_col}"
            df[datetime_col] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce",
            )
            df = df.drop(columns=[date_col, time_col])
            date_col = datetime_col
        else:
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Set datetime as index
        df = df.set_index(date_col)
        df.index.name = "datetime"

    except Exception as e:
        logger.warning(f"Failed to process datetime columns in Dask partition: {e}")

    return df


def _clean_numeric_data_dask(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize numeric data for Dask partitions."""
    numeric_columns = ["open", "high", "low", "close", "volume"]

    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Remove rows with NaN in price columns
            if col != "volume" and df[col].isna().any():
                na_count = df[col].isna().sum()
                logger.debug(
                    f"Removed {na_count} NaN values from {col} column in Dask partition"
                )
                df = df.dropna(subset=[col])

    return df


def _apply_validation_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Apply data validation to Dask DataFrame using map_partitions.

    Args:
        ddf: Dask DataFrame

    Returns:
        dd.DataFrame: Validated Dask DataFrame
    """

    def validate_partition(df):
        """Validate a single partition."""
        # Apply data validation
        df, validation_report = validate_data(
            df,
            outlier_detection=True,
            outlier_method="iqr",
            outlier_threshold=1.5,
            missing_value_strategy="forward_fill",
        )
        return df

    # Clear divisions to ensure metadata doesn't interfere
    ddf = ddf.clear_divisions()

    # Apply validation to each partition without specifying meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress Dask warnings
        ddf = ddf.map_partitions(validate_partition)

    return ddf


def compute_with_progress(
    ddf: dd.DataFrame, show_progress: bool = True
) -> pd.DataFrame:
    """
    Compute Dask DataFrame with optional progress bar.

    Args:
        ddf: Dask DataFrame to compute
        show_progress: Whether to show progress bar

    Returns:
        pd.DataFrame: Computed pandas DataFrame
    """
    logger.info("Computing Dask DataFrame...")

    if show_progress:
        try:
            # Use Dask's built-in progress bar
            with dask.config.set(scheduler="synchronous"):
                with ProgressBar():
                    result = ddf.compute()
        except Exception as e:
            logger.warning(f"Failed to use Dask progress bar: {e}")
            # Try computing without progress bar
            try:
                result = ddf.compute()
            except Exception as e2:
                logger.error(f"Dask computation failed: {e2}")
                # Try with clear_divisions to reset metadata
                try:
                    ddf_cleared = ddf.clear_divisions()
                    result = ddf_cleared.compute()
                except Exception as e3:
                    raise RuntimeError(
                        f"Dask computation failed after multiple attempts: {e3}"
                    ) from e3
    else:
        try:
            result = ddf.compute()
        except Exception as e:
            logger.error(f"Dask computation failed: {e}")
            # Try with clear_divisions to reset metadata
            try:
                ddf_cleared = ddf.clear_divisions()
                result = ddf_cleared.compute()
            except Exception as e2:
                raise RuntimeError(f"Dask computation failed: {e2}") from e2

    logger.info(
        f"Computation completed: {len(result)} rows, {len(result.columns)} columns"
    )
    return result


def monitor_dask_cluster() -> Dict[str, Any]:
    """
    Monitor Dask cluster status and resources.

    Returns:
        Dict with cluster information
    """
    try:
        from dask.distributed import Client

        client = Client()

        cluster_info = {
            "scheduler_address": client.scheduler_address,
            "workers": len(client.scheduler_info()["workers"]),
            "total_cores": sum(
                worker.get("ncores", 0)
                for worker in client.scheduler_info()["workers"].values()
            ),
            "total_memory": sum(
                worker.get("memory_limit", 0)
                for worker in client.scheduler_info()["workers"].values()
            ),
            "dashboard_link": client.dashboard_link,
        }

        return cluster_info

    except Exception as e:
        logger.warning(f"Failed to monitor Dask cluster: {e}")
        return {"error": str(e)}


def optimize_dask_performance(
    scheduler: str = "threads",
    memory_limit: str = "2GB",
    pool_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Optimize Dask performance settings.

    Args:
        scheduler: Scheduler type
        memory_limit: Memory limit per worker
        pool_size: Thread/process pool size

    Returns:
        Dict with optimization settings applied
    """
    settings = {}

    # Configure scheduler
    if scheduler == "threads":
        import multiprocessing

        if pool_size is None:
            pool_size = min(multiprocessing.cpu_count(), 8)

        dask.config.set(pool=pool_size)
        settings["pool_size"] = pool_size
        logger.info(f"Configured thread pool with {pool_size} workers")

    # Configure memory limits
    dask.config.set({"distributed.worker.memory.target": False})
    dask.config.set({"distributed.worker.memory.spill": False})

    settings["memory_limit"] = memory_limit
    logger.info(f"Configured memory limit: {memory_limit}")

    # Optimize for performance
    dask.config.set({"optimization.fuse.active": True})

    return settings


def benchmark_dask_engines(
    csv_path: str, chunk_sizes: List[int] = None, schedulers: List[str] = None
) -> Dict[str, Any]:
    """
    Benchmark different Dask configurations for performance testing.

    Args:
        csv_path: Path to test CSV file
        chunk_sizes: List of chunk sizes to test
        schedulers: List of schedulers to test

    Returns:
        Dict with benchmark results
    """
    if schedulers is None:
        schedulers = ["threads", "processes"]
    if chunk_sizes is None:
        chunk_sizes = [1000, 5000, 10000]
    logger.info("Starting Dask benchmarking...")

    results = {}

    for scheduler in schedulers:
        results[scheduler] = {}

        for chunk_size in chunk_sizes:
            logger.info(f"Testing {scheduler} scheduler with chunk_size={chunk_size}")

            try:
                start_time = time.time()

                # Create temporary config
                config = ProcessingConfig(engine_type="dask", chunk_size=chunk_size)

                # Process with Dask
                ddf = process_dask(
                    csv_path, config, scheduler=scheduler, show_progress=False
                )

                # Compute the result
                result = ddf.compute()

                end_time = time.time()
                processing_time = end_time - start_time

                # Record results
                results[scheduler][chunk_size] = {
                    "time": processing_time,
                    "rows": len(result),
                    "columns": len(result.columns),
                    "memory_mb": result.memory_usage(deep=True).sum() / (1024 * 1024),
                }

                logger.info(
                    f"  {scheduler}: {chunk_size} chunks - {processing_time:.2f}s, "
                    f"{len(result)} rows, {results[scheduler][chunk_size]['memory_mb']:.2f}MB"
                )

            except Exception as e:
                logger.error(f"  {scheduler}: {chunk_size} chunks - FAILED: {e}")
                results[scheduler][chunk_size] = {"error": str(e)}

    return results


def get_dask_cluster_info() -> Dict[str, Any]:
    """
    Get comprehensive Dask cluster information.

    Returns:
        Dict with detailed cluster information
    """
    try:
        from dask.distributed import Client

        client = Client()

        # Get scheduler info
        scheduler_info = client.scheduler_info()
        workers_info = scheduler_info.get("workers", {})

        cluster_info = {
            "scheduler_address": client.scheduler_address,
            "dashboard_link": client.dashboard_link,
            "version": dask.__version__,
            "workers": {},
            "total_cores": 0,
            "total_memory": 0,
            "total_threads": 0,
        }

        # Collect worker information
        for worker_addr, worker_info in workers_info.items():
            worker_data = {
                "address": worker_addr,
                "ncores": worker_info.get("ncores", 0),
                "memory_limit": worker_info.get("memory_limit", 0),
                "memory_used": worker_info.get("memory", 0),
                "metrics": worker_info.get("metrics", {}),
                "services": worker_info.get("services", {}),
                "status": worker_info.get("status", "unknown"),
            }

            cluster_info["workers"][worker_addr] = worker_data
            cluster_info["total_cores"] += worker_data["ncores"]
            cluster_info["total_memory"] += worker_data["memory_limit"]
            cluster_info["total_threads"] += worker_data["ncores"]

        return cluster_info

    except Exception as e:
        logger.error(f"Failed to get Dask cluster info: {e}")
        return {"error": str(e)}
