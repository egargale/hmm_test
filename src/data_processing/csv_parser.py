"""
CSV Parser Module

Provides robust CSV parsing functionality for multi-format OHLCV data,
including datetime detection, column standardization, and memory-efficient processing.
"""

import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


def process_csv(
    file_path: Union[str, Path],
    date_columns: Optional[List[str]] = None,
    time_columns: Optional[List[str]] = None,
    symbol_filter: Optional[str] = None,
    chunk_size: Optional[int] = None,
    memory_limit_gb: float = 8.0,
    downcast_floats: bool = True,
    downcast_ints: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Process and parse multi-format OHLCV CSV files with memory efficiency.

    Args:
        file_path: Path to the CSV file
        date_columns: List of potential date column names (auto-detected if None)
        time_columns: List of potential time column names (auto-detected if None)
        symbol_filter: Optional symbol name to filter data
        chunk_size: Optional chunk size for memory-efficient processing
        memory_limit_gb: Memory limit in GB for processing
        downcast_floats: Whether to downcast float64 to float32
        downcast_ints: Whether to downcast int64 to int32
        **kwargs: Additional arguments passed to pandas.read_csv

    Returns:
        pd.DataFrame: Processed OHLCV data with standardized columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required OHLCV columns are missing
        MemoryError: If processing exceeds memory limit
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.info(f"Processing CSV file: {file_path}")

    # Auto-detect date/time columns if not provided
    if date_columns is None:
        date_columns = [
            "date",
            "Date",
            "datetime",
            "DateTime",
            "timestamp",
            "Timestamp",
        ]

    if time_columns is None:
        time_columns = [
            "time",
            "Time",
            "datetime",
            "DateTime",
            "timestamp",
            "Timestamp",
        ]

    # Determine if we need chunked processing
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    use_chunking = (
        chunk_size is not None or file_size_mb > 100
    )  # Use chunking for files > 100MB

    if use_chunking:
        logger.info(f"Using chunked processing (file size: {file_size_mb:.1f}MB)")
        return _process_chunked(
            file_path,
            date_columns,
            time_columns,
            symbol_filter,
            chunk_size,
            memory_limit_gb,
            downcast_floats,
            downcast_ints,
            **kwargs,
        )
    else:
        logger.info(f"Using standard processing (file size: {file_size_mb:.1f}MB)")
        return _process_standard(
            file_path,
            date_columns,
            time_columns,
            symbol_filter,
            downcast_floats,
            downcast_ints,
            **kwargs,
        )


def _process_standard(
    file_path: Path,
    date_columns: List[str],
    time_columns: List[str],
    symbol_filter: Optional[str],
    downcast_floats: bool,
    downcast_ints: bool,
    **kwargs,
) -> pd.DataFrame:
    """Process CSV file using standard pandas read_csv."""

    # First, read a small sample to understand the structure
    sample_df = pd.read_csv(file_path, nrows=5)
    logger.debug(f"CSV columns detected: {list(sample_df.columns)}")

    # Detect delimiter if not specified
    if "sep" not in kwargs:
        sample_text = file_path.read_text(encoding="utf-8")[:1000]
        if ";" in sample_text and "," not in sample_text[:100]:
            kwargs["sep"] = ";"
        else:
            kwargs["sep"] = ","

    # Read the full file
    df = pd.read_csv(file_path, **kwargs)

    # Process the dataframe
    df = _standardize_dataframe(
        df, date_columns, time_columns, symbol_filter, downcast_floats, downcast_ints
    )

    logger.info(f"Processed {len(df)} rows of data")
    return df


def _process_chunked(
    file_path: Path,
    date_columns: List[str],
    time_columns: List[str],
    symbol_filter: Optional[str],
    chunk_size: Optional[int],
    memory_limit_gb: float,
    downcast_floats: bool,
    downcast_ints: bool,
    **kwargs,
) -> pd.DataFrame:
    """Process CSV file in chunks for memory efficiency."""

    # Set default chunk size if not provided
    if chunk_size is None:
        chunk_size = 10000

    # First, read a small sample to understand the structure
    sample_df = pd.read_csv(file_path, nrows=5)
    logger.debug(f"CSV columns detected: {list(sample_df.columns)}")

    # Detect delimiter if not specified
    if "sep" not in kwargs:
        sample_text = file_path.read_text(encoding="utf-8")[:1000]
        if ";" in sample_text and "," not in sample_text[:100]:
            kwargs["sep"] = ";"
        else:
            kwargs["sep"] = ","

    # Process chunks
    chunks = []
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, **kwargs)):
        logger.debug(f"Processing chunk {i + 1} ({len(chunk)} rows)")

        # Standardize the chunk
        chunk = _standardize_dataframe(
            chunk,
            date_columns,
            time_columns,
            symbol_filter,
            downcast_floats,
            downcast_ints,
        )

        chunks.append(chunk)
        total_rows += len(chunk)

        # Memory management
        if i % 10 == 0:  # Every 10 chunks
            gc.collect()

            # Estimate memory usage
            current_memory = sum(
                len(chunk) * chunk.memory_usage(deep=True).sum() for chunk in chunks
            ) / (1024**3)

            if current_memory > memory_limit_gb * 0.8:  # 80% of limit
                logger.warning(
                    f"Memory usage approaching limit: {current_memory:.2f}GB"
                )

    # Combine all chunks
    logger.info(f"Combining {len(chunks)} chunks ({total_rows} total rows)")
    result_df = pd.concat(chunks, ignore_index=True)

    # Clean up
    del chunks
    gc.collect()

    return result_df


def _standardize_dataframe(
    df: pd.DataFrame,
    date_columns: List[str],
    time_columns: List[str],
    symbol_filter: Optional[str],
    downcast_floats: bool,
    downcast_ints: bool,
) -> pd.DataFrame:
    """Standardize dataframe column names, types, and format."""

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Filter by symbol if specified
    if symbol_filter is not None and "symbol" in df.columns:
        original_len = len(df)
        df = df[df["symbol"].str.contains(symbol_filter, case=False, na=False)]
        logger.info(
            f"Filtered to {len(df)} rows by symbol '{symbol_filter}' "
            f"(from {original_len} rows)"
        )

    # Detect and standardize OHLCV columns
    column_mapping = _detect_ohlcv_columns(df)

    if not column_mapping:
        raise ValueError(
            "Could not detect required OHLCV columns. "
            "Expected columns: open, high, low, close, volume (or similar)"
        )

    # Rename columns to standard names
    df = df.rename(columns=column_mapping)

    # Process datetime columns
    df = _process_datetime_columns(df, date_columns, time_columns)

    # Ensure required columns exist
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Clean numeric data
    df = _clean_numeric_data(df)

    # Downcast numeric types for memory efficiency
    if downcast_floats:
        df = _downcast_floats(df)

    if downcast_ints:
        df = _downcast_ints(df)

    # Sort by datetime index if available
    if hasattr(df.index, "to_pydatetime"):
        df = df.sort_index()

    # Validate OHLCV data integrity
    _validate_ohlcv_data(df)

    return df


def _detect_ohlcv_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect OHLCV columns with various naming conventions."""

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

    logger.debug(f"Detected column mapping: {column_mapping}")
    return column_mapping


def _process_datetime_columns(
    df: pd.DataFrame, date_columns: List[str], time_columns: List[str]
) -> pd.DataFrame:
    """Process and combine date/time columns into datetime index."""

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
        logger.warning("No date column found, keeping original index")
        return df

    try:
        if time_col and time_col in df.columns:
            # Combine date and time columns
            datetime_col = f"{date_col}_{time_col}"
            df[datetime_col] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                infer_datetime_format=True,
                errors="coerce",
            )
            df = df.drop(columns=[date_col, time_col])
            date_col = datetime_col
        else:
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(
                df[date_col], infer_datetime_format=True, errors="coerce"
            )

        # Set datetime as index
        df = df.set_index(date_col)
        df.index.name = "datetime"

        logger.info(f"Set datetime index with {len(df)} rows")

    except Exception as e:
        logger.warning(f"Failed to process datetime columns: {e}")
        logger.warning("Keeping original index")

    return df


def _clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize numeric data."""

    numeric_columns = ["open", "high", "low", "close", "volume"]

    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Remove rows with NaN in price columns
            if col != "volume" and df[col].isna().any():
                na_count = df[col].isna().sum()
                logger.warning(f"Found {na_count} NaN values in {col} column")
                df = df.dropna(subset=[col])

    return df


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 for memory efficiency."""

    float_columns = df.select_dtypes(include=["float64"]).columns

    for col in float_columns:
        if df[col].dtype == "float64":
            # Check if we can safely downcast to float32
            min_val = df[col].min()
            max_val = df[col].max()

            if (
                np.finfo(np.float32).min <= min_val
                and max_val <= np.finfo(np.float32).max
            ):
                df[col] = df[col].astype("float32")
                logger.debug(f"Downcasted {col} to float32")

    return df


def _downcast_ints(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast int64 columns to smaller integer types for memory efficiency."""

    int_columns = df.select_dtypes(include=["int64"]).columns

    for col in int_columns:
        if df[col].dtype == "int64":
            # Find the smallest integer type that can hold the data
            min_val = df[col].min()
            max_val = df[col].max()

            if np.iinfo(np.int8).min <= min_val and max_val <= np.iinfo(np.int8).max:
                df[col] = df[col].astype("int8")
                logger.debug(f"Downcasted {col} to int8")
            elif (
                np.iinfo(np.int16).min <= min_val and max_val <= np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype("int16")
                logger.debug(f"Downcasted {col} to int16")
            elif (
                np.iinfo(np.int32).min <= min_val and max_val <= np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype("int32")
                logger.debug(f"Downcasted {col} to int32")

    return df


def _validate_ohlcv_data(df: pd.DataFrame) -> None:
    """Validate OHLCV data integrity."""

    # Check for negative prices
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if col in df.columns and (df[col] <= 0).any():
            negative_count = (df[col] <= 0).sum()
            logger.warning(
                f"Found {negative_count} non-positive values in {col} column"
            )

    # Check for negative volume
    if "volume" in df.columns and (df["volume"] < 0).any():
        negative_count = (df["volume"] < 0).sum()
        logger.warning(f"Found {negative_count} negative values in volume column")

    # Check OHLC consistency (high should be >= open,close and low should be <= open,close)
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        inconsistent_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        inconsistent_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()

        if inconsistent_high > 0:
            logger.warning(
                f"Found {inconsistent_high} rows where high < max(open,close)"
            )

        if inconsistent_low > 0:
            logger.warning(f"Found {inconsistent_low} rows where low > min(open,close)")

    logger.info("OHLCV data validation completed")


def get_csv_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a CSV file without loading it completely.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dict containing file information
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    info = {
        "file_path": str(file_path),
        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime),
    }

    # Read first few rows to get column info
    try:
        sample_df = pd.read_csv(file_path, nrows=5)
        info.update(
            {
                "columns": list(sample_df.columns),
                "num_columns": len(sample_df.columns),
                "sample_data": sample_df.head(3).to_dict("records"),
            }
        )

        # Try to estimate total rows
        if info["file_size_mb"] < 100:  # Only for small files
            try:
                full_df = pd.read_csv(file_path)
                info["estimated_rows"] = len(full_df)
                info["memory_usage_mb"] = full_df.memory_usage(deep=True).sum() / (
                    1024 * 1024
                )
            except MemoryError:
                info["estimated_rows"] = "Unknown (too large to load)"
        else:
            info["estimated_rows"] = "Unknown (large file)"

    except Exception as e:
        logger.warning(f"Could not analyze CSV structure: {e}")
        info["columns"] = []
        info["num_columns"] = 0

    return info
