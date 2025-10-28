"""
Streaming Engine Module

Implements memory-efficient chunked processing using pandas with configurable
chunk sizes, progress monitoring, and automatic memory management.
"""

import gc
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import psutil
from tqdm import tqdm

from data_processing import add_features, validate_data
from utils import ProcessingConfig, get_logger

logger = get_logger(__name__)


def process_streaming(
    csv_path: str,
    config: ProcessingConfig,
    chunk_size: Optional[int] = None,
    memory_limit_gb: float = 8.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Process CSV file using streaming chunks for memory efficiency.

    Args:
        csv_path: Path to the CSV file
        config: Processing configuration
        chunk_size: Optional chunk size (auto-detected if None)
        memory_limit_gb: Memory limit in GB for processing
        show_progress: Whether to show progress bar

    Returns:
        pd.DataFrame: Fully processed DataFrame

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        MemoryError: If memory limit is exceeded
        ValueError: If configuration is invalid
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Starting streaming processing of {csv_path}")
    logger.info(
        f"Configuration: engine={config.engine_type}, chunk_size={chunk_size}, memory_limit={memory_limit_gb}GB"
    )

    # Get file info for auto-detection
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")

    # Auto-detect chunk size if not provided
    if chunk_size is None:
        chunk_size = _auto_detect_chunk_size(file_size_mb, memory_limit_gb)
        logger.info(f"Auto-detected chunk size: {chunk_size}")

    # Initialize processing variables
    processed_chunks = []
    total_rows = 0
    processing_times = []
    memory_usage_peak = 0.0

    # Monitor initial memory usage
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

    try:
        # Create progress bar
        pbar = None
        if show_progress:
            # Estimate total chunks for progress bar
            estimated_chunks = max(
                1, (file_size_mb * 1024 * 1024) // (chunk_size * 1000)
            )  # Rough estimate
            pbar = tqdm(
                total=estimated_chunks,
                desc="Processing chunks",
                unit="chunks",
                dynamic_ncols=True,
            )

        # Process file in chunks
        chunk_reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype={
                "open": "float32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "volume": "float32",
            },
        )

        for i, chunk in enumerate(chunk_reader):
            chunk_start_time = time.time()

            # Monitor memory before processing chunk
            psutil.Process().memory_info().rss / (1024 * 1024 * 1024)

            # Process chunk
            try:
                # First, standardize the chunk using csv_parser logic
                chunk = _standardize_chunk(chunk)

                # Apply feature engineering
                chunk = add_features(
                    chunk, indicator_config=config.indicators, downcast_floats=True
                )

                # Apply data validation
                chunk, validation_report = validate_data(
                    chunk,
                    outlier_detection=True,
                    outlier_method="iqr",
                    outlier_threshold=1.5,
                    missing_value_strategy="forward_fill",
                )

                # Store processed chunk
                processed_chunks.append(chunk)
                total_rows += len(chunk)

                # Log chunk processing info
                chunk_processing_time = time.time() - chunk_start_time
                processing_times.append(chunk_processing_time)

                logger.debug(
                    f"Chunk {i + 1}: {len(chunk)} rows, "
                    f"{chunk_processing_time:.2f}s, "
                    f"{chunk.columns.tolist()[:5]}... columns"
                )

            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                raise

            # Memory management
            del chunk
            gc.collect()

            # Monitor memory after processing
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
            memory_usage_peak = max(memory_usage_peak, memory_after)

            # Check memory limit
            if memory_after > memory_limit_gb:
                logger.warning(
                    f"Memory usage ({memory_after:.2f}GB) approaching limit ({memory_limit_gb}GB)"
                )
                if memory_after > memory_limit_gb * 1.1:  # 10% buffer
                    raise MemoryError(
                        f"Memory limit exceeded: {memory_after:.2f}GB > {memory_limit_gb}GB"
                    )

            # Update progress bar
            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "rows": total_rows,
                        "memory": f"{memory_after:.2f}GB",
                        "time": f"{chunk_processing_time:.1f}s",
                    }
                )

        # Close progress bar
        if pbar:
            pbar.close()

        # Combine all chunks
        logger.info(f"Combining {len(processed_chunks)} processed chunks...")
        combine_start_time = time.time()

        final_df = pd.concat(processed_chunks, ignore_index=True)

        combine_time = time.time() - combine_start_time
        logger.info(f"Combined {len(final_df)} rows in {combine_time:.2f}s")

        # Clean up memory
        del processed_chunks
        gc.collect()

        # Final memory check
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
        memory_used = final_memory - initial_memory

        # Log processing summary
        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        total_processing_time = sum(processing_times)

        logger.info("Streaming processing completed successfully:")
        logger.info(f"  - Total rows processed: {len(final_df)}")
        logger.info(f"  - Total chunks: {len(processing_times)}")
        logger.info(f"  - Total processing time: {total_processing_time:.2f}s")
        logger.info(f"  - Average chunk time: {avg_processing_time:.2f}s")
        logger.info(f"  - Memory used: {memory_used:.2f}GB")
        logger.info(f"  - Peak memory: {memory_usage_peak:.2f}GB")
        logger.info(f"  - Final columns: {len(final_df.columns)}")

        return final_df

    except Exception as e:
        logger.error(f"Streaming processing failed: {e}")
        raise


def _auto_detect_chunk_size(file_size_mb: float, memory_limit_gb: float) -> int:
    """
    Auto-detect optimal chunk size based on file size and memory limit.

    Args:
        file_size_mb: File size in MB
        memory_limit_gb: Memory limit in GB

    Returns:
        int: Recommended chunk size
    """
    memory_limit_mb = memory_limit_gb * 1024

    # Reserve 50% of memory limit for safety
    safe_memory_mb = memory_limit_mb * 0.5

    # Base chunk size estimation
    # Assuming each row needs ~1KB of memory with features
    estimated_row_size_kb = 1.0

    # Calculate chunk size based on available memory
    chunk_size = int((safe_memory_mb * 1024) / estimated_row_size_kb)

    # Apply practical limits
    if file_size_mb < 10:  # Small files
        chunk_size = min(chunk_size, 1000)
    elif file_size_mb < 100:  # Medium files
        chunk_size = min(chunk_size, 5000)
    elif file_size_mb < 1000:  # Large files
        chunk_size = min(chunk_size, 10000)
    else:  # Very large files
        chunk_size = min(chunk_size, 20000)

    # Ensure minimum chunk size
    chunk_size = max(chunk_size, 100)

    # Round to nearest 100 for cleaner numbers
    chunk_size = round(chunk_size / 100) * 100

    logger.debug(
        f"Auto-detected chunk size: {chunk_size} (file: {file_size_mb:.1f}MB, limit: {memory_limit_gb}GB)"
    )

    return chunk_size


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with memory usage information in GB
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss": memory_info.rss / (1024**3),  # Resident Set Size
        "vms": memory_info.vms / (1024**3),  # Virtual Memory Size
        "percent": process.memory_percent(),  # Memory usage percentage
        "available": psutil.virtual_memory().available / (1024**3),  # Available memory
    }


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage during function execution.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with memory monitoring
    """

    def wrapper(*args, **kwargs):
        # Memory before
        memory_before = get_memory_usage()
        logger.debug(f"Memory before {func.__name__}: {memory_before['rss']:.2f}GB")

        try:
            result = func(*args, **kwargs)

            # Memory after
            memory_after = get_memory_usage()
            memory_diff = memory_after["rss"] - memory_before["rss"]

            logger.debug(
                f"Memory after {func.__name__}: {memory_after['rss']:.2f}GB (Δ{memory_diff:+.2f}GB)"
            )

            return result

        except Exception:
            memory_error = get_memory_usage()
            logger.error(
                f"Memory during {func.__name__} error: {memory_error['rss']:.2f}GB"
            )
            raise

    return wrapper


def optimize_chunk_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame chunk.

    Args:
        df: DataFrame to optimize

    Returns:
        Optimized DataFrame
    """
    # Downcast float64 to float32
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype("float32")

    # Downcast int64 to smallest possible type
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        min_val = df[col].min()
        max_val = df[col].max()

        if min_val >= 0:  # Unsigned integers
            if max_val < 255:
                df[col] = df[col].astype("uint8")
            elif max_val < 65535:
                df[col] = df[col].astype("uint16")
            elif max_val < 4294967295:
                df[col] = df[col].astype("uint32")
        else:  # Signed integers
            if min_val >= -128 and max_val < 127:
                df[col] = df[col].astype("int8")
            elif min_val >= -32768 and max_val < 32767:
                df[col] = df[col].astype("int16")
            elif min_val >= -2147483648 and max_val < 2147483647:
                df[col] = df[col].astype("int32")

    return df


def estimate_processing_time(
    csv_path: str, chunk_size: int = 1000, sample_chunks: int = 3
) -> Dict[str, float]:
    """
    Estimate processing time by processing a few sample chunks.

    Args:
        csv_path: Path to CSV file
        chunk_size: Chunk size to use for estimation
        sample_chunks: Number of chunks to sample

    Returns:
        Dict with time estimates
    """
    logger.info(f"Estimating processing time using {sample_chunks} sample chunks...")

    processing_times = []

    try:
        chunk_reader = pd.read_csv(csv_path, chunksize=chunk_size)

        for i, chunk in enumerate(chunk_reader):
            if i >= sample_chunks:
                break

            start_time = time.time()

            # Process chunk (simplified version for estimation)
            chunk = optimize_chunk_memory(chunk)
            chunk = add_features(chunk, downcast_floats=True)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            del chunk
            gc.collect()

        if processing_times:
            avg_time_per_chunk = sum(processing_times) / len(processing_times)

            # Estimate file characteristics
            file_size_mb = Path(csv_path).stat().st_size / (1024 * 1024)

            # Rough estimate of total chunks
            estimated_chunks = int(
                file_size_mb * 1024 * 1024 / (chunk_size * 1000)
            )  # Very rough

            estimated_total_time = avg_time_per_chunk * estimated_chunks

            return {
                "avg_time_per_chunk": avg_time_per_chunk,
                "estimated_chunks": estimated_chunks,
                "estimated_total_time": estimated_total_time,
                "sample_chunks_processed": len(processing_times),
            }
        else:
            return {"error": "No chunks processed for estimation"}

    except Exception as e:
        logger.error(f"Error during time estimation: {e}")
        return {"error": str(e)}


def _standardize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a chunk of data similar to csv_parser._standardize_dataframe.

    Args:
        df: Raw DataFrame chunk

    Returns:
        Standardized DataFrame with proper OHLCV columns and datetime index
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Detect and standardize OHLCV columns
    column_mapping = _detect_ohlcv_columns_chunk(df)

    if not column_mapping:
        raise ValueError("Could not detect required OHLCV columns in chunk")

    # Rename columns to standard names
    df = df.rename(columns=column_mapping)

    # Process datetime columns
    df = _process_datetime_columns_chunk(df)

    # Ensure required columns exist
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    # Clean numeric data
    df = _clean_numeric_data_chunk(df)

    return df


def _detect_ohlcv_columns_chunk(df: pd.DataFrame) -> Dict[str, str]:
    """Detect OHLCV columns with various naming conventions for a chunk."""
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


def _process_datetime_columns_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Process and combine date/time columns into datetime index for a chunk."""
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
        logger.warning(f"Failed to process datetime columns in chunk: {e}")

    return df


def _clean_numeric_data_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize numeric data for a chunk."""
    numeric_columns = ["open", "high", "low", "close", "volume"]

    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Remove rows with NaN in price columns
            if col != "volume" and df[col].isna().any():
                na_count = df[col].isna().sum()
                logger.debug(
                    f"Removed {na_count} NaN values from {col} column in chunk"
                )
                df = df.dropna(subset=[col])

    return df
