"""
Enhanced streaming data processor for large datasets.

This module provides streaming processing capabilities for large CSV files,
with memory optimization, progress tracking, and error recovery.
"""

import asyncio
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from utils.data_types import CSVFormat
from utils.logging_config import get_logger
from .feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class DataProcessingError(Exception):
    """Data processing specific errors"""
    pass


class StreamingDataProcessor:
    """
    Enhanced streaming processor for large datasets.

    This class provides memory-efficient processing of large CSV files
    by processing data in chunks with optional parallel processing.
    """

    def __init__(self, config):
        """
        Initialize the streaming processor.

        Args:
            config: Streaming configuration object
        """
        self.config = config
        self.chunk_size = config.chunk_size
        self.max_memory_mb = config.max_memory_mb
        self.processing_mode = config.processing_mode

        # Processing state
        self.total_rows_processed = 0
        self.memory_usage_mb = 0.0
        self.start_time = None

        logger.info(f"Initialized streaming processor: chunk_size={self.chunk_size}, mode={self.processing_mode.value}")

    async def process_stream(
        self,
        csv_path: Path,
        feature_engineer: FeatureEngineer
    ) -> pd.DataFrame:
        """
        Process large CSV files in streaming fashion.

        Args:
            csv_path: Path to CSV file
            feature_engineer: Feature engineering instance

        Returns:
            Processed dataframe with features
        """
        try:
            self.start_time = datetime.now()
            logger.info(f"Starting stream processing: {csv_path}")

            # Validate CSV format
            csv_format = self._validate_csv_format(csv_path)

            # Determine processing strategy
            should_stream = self._should_use_streaming(csv_path)

            if should_stream:
                # Process in chunks
                result = await self._process_chunks(csv_path, feature_engineer, csv_format)
            else:
                # Process entire file at once
                result = await self._process_entire_file(csv_path, feature_engineer, csv_format)

            # Log processing statistics
            self._log_processing_stats(len(result))

            return result

        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise DataProcessingError(f"Stream processing failed: {e}")

    def _validate_csv_format(self, csv_path: Path) -> CSVFormat:
        """
        Validate CSV format and detect column structure.

        Args:
            csv_path: Path to CSV file

        Returns:
            Detected CSV format
        """
        try:
            logger.info("Validating CSV format...")

            # Read first few rows to validate format
            first_chunk = pd.read_csv(csv_path, nrows=5)

            # Strip whitespace from column names
            first_chunk.columns = first_chunk.columns.str.strip()

            # Detect format type
            if "DateTime" in first_chunk.columns:
                format_type = "datetime"
                required_columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
            elif "Date" in first_chunk.columns and "Time" in first_chunk.columns:
                format_type = "date_time"
                required_columns = ["Date", "Time", "Open", "High", "Low", "Last", "Volume"]
            else:
                raise ValueError("Unrecognized CSV format")

            # Check required columns
            missing_columns = [col for col in required_columns if col not in first_chunk.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            csv_format = CSVFormat(
                format_type=format_type,
                columns=first_chunk.columns.tolist(),
                required_columns=required_columns
            )

            logger.info(f"CSV format detected: {format_type}")
            return csv_format

        except Exception as e:
            raise DataProcessingError(f"CSV format validation failed: {e}")

    def _should_use_streaming(self, csv_path: Path) -> bool:
        """
        Determine if streaming should be used based on file size and configuration.

        Args:
            csv_path: Path to CSV file

        Returns:
            True if streaming should be used
        """
        if self.processing_mode.value == "streaming":
            return True
        elif self.processing_mode.value == "memory":
            return False
        else:  # hybrid mode
            # Check file size
            file_size_mb = csv_path.stat().st_size / (1024 * 1024)
            return file_size_mb > self.max_memory_mb

    async def _process_chunks(
        self,
        csv_path: Path,
        feature_engineer: FeatureEngineer,
        csv_format: CSVFormat
    ) -> pd.DataFrame:
        """
        Process CSV file in chunks.

        Args:
            csv_path: Path to CSV file
            feature_engineer: Feature engineering instance
            csv_format: CSV format information

        Returns:
            Processed dataframe
        """
        logger.info("Processing CSV in chunks...")

        # Initialize processing
        frames = []
        chunk_iterator = self._create_chunk_iterator(csv_path, csv_format)

        # Process chunks with progress bar
        if self.config.show_progress:
            pbar = tqdm(desc="Processing chunks", unit="chunks")
        else:
            pbar = None

        async for chunk_num, chunk in chunk_iterator:
            try:
                # Process individual chunk
                processed_chunk = await self._process_chunk(
                    chunk, feature_engineer, chunk_num
                )

                if processed_chunk is not None and len(processed_chunk) > 0:
                    frames.append(processed_chunk)
                    self.total_rows_processed += len(processed_chunk)

                # Update progress
                if pbar:
                    pbar.update(1)

                # Memory management
                if chunk_num % self.config.memory_cleanup_interval == 0:
                    await self._cleanup_memory()

            except Exception as e:
                if self.config.skip_errors:
                    logger.warning(f"Skipping chunk {chunk_num} due to error: {e}")
                    continue
                else:
                    raise DataProcessingError(f"Failed to process chunk {chunk_num}: {e}")

        if pbar:
            pbar.close()

        # Combine all processed chunks
        if not frames:
            raise DataProcessingError("No chunks were successfully processed")

        logger.info(f"Combining {len(frames)} processed chunks...")
        result = pd.concat(frames, ignore_index=False).sort_index()

        return result

    async def _process_entire_file(
        self,
        csv_path: Path,
        feature_engineer: FeatureEngineer,
        csv_format: CSVFormat
    ) -> pd.DataFrame:
        """
        Process entire CSV file at once.

        Args:
            csv_path: Path to CSV file
            feature_engineer: Feature engineering instance
            csv_format: CSV format information

        Returns:
            Processed dataframe
        """
        logger.info("Processing entire file at once...")

        # Load entire file
        df = pd.read_csv(csv_path)

        # Preprocess dataframe
        df = self._preprocess_dataframe(df, csv_format)

        # Apply feature engineering
        result = feature_engineer.add_features(df)

        self.total_rows_processed = len(result)
        return result

    def _create_chunk_iterator(
        self,
        csv_path: Path,
        csv_format: CSVFormat
    ) -> Iterator[Tuple[int, pd.DataFrame]]:
        """
        Create iterator for reading CSV in chunks.

        Args:
            csv_path: Path to CSV file
            csv_format: CSV format information

        Yields:
            Tuple of (chunk_number, chunk_dataframe)
        """
        chunk_reader = pd.read_csv(
            csv_path,
            chunksize=self.chunk_size,
            dtype=self._get_dtypes()
        )

        yield from enumerate(chunk_reader, 1)

    def _get_dtypes(self) -> Dict[str, str]:
        """Get optimal data types for memory efficiency"""
        if self.config.downcast_dtypes:
            return {
                "Open": "float32",
                "High": "float32",
                "Low": "float32",
                "Close": "float32",
                "Last": "float32",
                "Volume": "float32"
            }
        return {}

    async def _process_chunk(
        self,
        chunk: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        chunk_num: int
    ) -> Optional[pd.DataFrame]:
        """
        Process individual chunk with feature engineering.

        Args:
            chunk: Input chunk dataframe
            feature_engineer: Feature engineering instance
            chunk_num: Chunk number for logging

        Returns:
            Processed chunk or None if processing failed
        """
        try:
            # Retry logic
            for attempt in range(self.config.max_retries):
                try:
                    # Preprocess chunk
                    chunk = self._preprocess_chunk(chunk, chunk_num)

                    # Apply feature engineering
                    processed_chunk = feature_engineer.add_features(chunk)

                    return processed_chunk

                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} failed: {e}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_num}: {e}")
            raise

    def _preprocess_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> pd.DataFrame:
        """
        Preprocess individual chunk.

        Args:
            chunk: Input chunk
            chunk_num: Chunk number

        Returns:
            Preprocessed chunk
        """
        # Strip whitespace from column names
        chunk.columns = chunk.columns.str.strip()

        # Handle datetime parsing
        if "DateTime" in chunk.columns:
            chunk["DateTime"] = pd.to_datetime(chunk["DateTime"])
            chunk = chunk.set_index("DateTime")
        elif "Date" in chunk.columns and "Time" in chunk.columns:
            chunk["DateTime"] = pd.to_datetime(chunk["Date"] + " " + chunk["Time"])
            chunk = chunk.set_index("DateTime").drop(["Date", "Time"], axis=1)

        # Rename 'Last' to 'Close' if needed
        if "Last" in chunk.columns and "Close" not in chunk.columns:
            chunk = chunk.rename(columns={"Last": "Close"})

        # Handle columns with leading spaces (fallback)
        rename_dict = {}
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if f" {col}" in chunk.columns and col not in chunk.columns:
                rename_dict[f" {col}"] = col
        if rename_dict:
            chunk = chunk.rename(columns=rename_dict)

        # Downcast dtypes for memory efficiency
        if self.config.downcast_dtypes:
            chunk = chunk.astype(self._get_dtypes())

        return chunk

    def _preprocess_dataframe(self, df: pd.DataFrame, csv_format: CSVFormat) -> pd.DataFrame:
        """
        Preprocess entire dataframe.

        Args:
            df: Input dataframe
            csv_format: CSV format information

        Returns:
            Preprocessed dataframe
        """
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Handle datetime parsing based on format
        if csv_format.format_type == "datetime":
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df = df.set_index("DateTime")
        elif csv_format.format_type == "date_time":
            df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
            df = df.set_index("DateTime").drop(["Date", "Time"], axis=1)

        # Rename 'Last' to 'Close' if needed
        if "Last" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Last": "Close"})

        # Handle columns with leading spaces
        rename_dict = {}
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if f" {col}" in df.columns and col not in df.columns:
                rename_dict[f" {col}"] = col
        if rename_dict:
            df = df.rename(columns=rename_dict)

        # Downcast dtypes for memory efficiency
        if self.config.downcast_dtypes:
            df = df.astype(self._get_dtypes())

        return df

    async def _cleanup_memory(self) -> None:
        """Perform memory cleanup"""
        gc.collect()
        # Update memory usage if possible
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass

    def _log_processing_stats(self, final_row_count: int) -> None:
        """Log processing statistics"""
        if self.start_time:
            processing_time = (datetime.now() - self.start_time).total_seconds()
            rows_per_second = final_row_count / processing_time if processing_time > 0 else 0

            logger.info(
                f"Processing completed: {final_row_count} rows in {processing_time:.2f}s "
                f"({rows_per_second:.0f} rows/sec)"
            )

            if self.memory_usage_mb > 0:
                logger.info(f"Peak memory usage: {self.memory_usage_mb:.1f} MB")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_rows_processed": self.total_rows_processed,
            "memory_usage_mb": self.memory_usage_mb,
            "chunk_size": self.chunk_size,
            "processing_mode": self.processing_mode.value
        }
