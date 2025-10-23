"""
Performance optimization engine for CSV processing.

This module provides advanced performance optimizations including parallel processing,
memory mapping, adaptive chunking, and vectorized operations.
"""

import os
import gc
import mmap
from typing import Callable, Any, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    # Processing
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 10000
    adaptive_chunking: bool = True

    # Memory
    memory_limit_mb: int = 1024
    enable_memory_mapping: bool = True
    downcast_dtypes: bool = True

    # I/O
    buffer_size: int = 65536  # 64KB
    enable_async_io: bool = True

    # Optimization
    cache_results: bool = False
    cache_size_mb: int = 100


@dataclass
class PerformanceMetrics:
    """Performance measurement metrics."""
    processing_time: float
    memory_usage_mb: float
    rows_processed: int
    rows_per_second: float
    cache_hit_rate: float
    parallel_efficiency: float


class PerformanceOptimizer:
    """
    Advanced performance optimization engine for CSV processing.

    Features:
    - Parallel processing with ThreadPool/ProcessPool
    - Memory mapping for large files
    - Adaptive chunk size optimization
    - Vectorized operations
    - Smart memory management
    - Result caching
    - Performance monitoring
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self._setup_performance_config()

        # Performance tracking
        self.metrics_cache = {}
        self.result_cache = {}

        # Memory tracking
        self.memory_usage_history = []

        logger.info(f"Performance optimizer initialized with config: "
                   f"parallel={self.config.enable_parallel_processing}, "
                   f"workers={self.config.max_workers}, "
                   f"chunk_size={self.config.chunk_size}")

    def _setup_performance_config(self) -> None:
        """Setup performance configuration based on system capabilities."""
        # Auto-detect optimal worker count
        if self.config.max_workers is None:
            cpu_count = mp.cpu_count()
            # Use fewer workers to avoid memory pressure
            self.config.max_workers = min(cpu_count, 4)
            logger.debug(f"Auto-detected optimal worker count: {self.config.max_workers}")

        # Adjust chunk size based on available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            if self.config.adaptive_chunking:
                # Larger chunk size for more memory
                if available_memory_gb > 8:
                    self.config.chunk_size = 50000
                elif available_memory_gb > 4:
                    self.config.chunk_size = 25000
                else:
                    self.config.chunk_size = 10000

                logger.debug(f"Adaptive chunk size set to {self.config.chunk_size} "
                           f"based on {available_memory_gb:.1f}GB available memory")
        except ImportError:
            logger.debug("psutil not available, using default chunk size")

    def optimize_chunk_size(self, file_size: int, available_memory: int) -> int:
        """
        Calculate optimal chunk size based on file size and available memory.

        Args:
            file_size: Size of file in bytes
            available_memory: Available memory in bytes

        Returns:
            Optimal chunk size in rows
        """
        if not self.config.adaptive_chunking:
            return self.config.chunk_size

        # Estimate row size (rough calculation)
        estimated_row_size = 100  # bytes per row (conservative estimate)

        # Calculate memory-based chunk size
        memory_based_rows = (available_memory * 0.5) / estimated_row_size

        # Calculate file-based chunk size (aim for 10-20 chunks total)
        file_based_rows = file_size / (estimated_row_size * 15)

        # Use the smaller of the two, with bounds
        optimal_rows = min(memory_based_rows, file_based_rows)
        optimal_rows = max(1000, min(optimal_rows, 100000))  # Bounds: 1K - 100K rows

        logger.debug(f"Optimized chunk size: {optimal_rows:.0f} rows "
                    f"(memory: {memory_based_rows:.0f}, file: {file_based_rows:.0f})")

        return int(optimal_rows)

    def apply_parallel_processing(self, processor_func: Callable, data: Any,
                                 use_processes: bool = False) -> Any:
        """
        Apply parallel processing to a function.

        Args:
            processor_func: Function to process data
            data: Data to process (should be iterable)
            use_processes: Use ProcessPool instead of ThreadPool

        Returns:
            Processed results
        """
        if not self.config.enable_parallel_processing or len(data) == 1:
            return [processor_func(item) for item in data]

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        try:
            with executor_class(max_workers=self.config.max_workers) as executor:
                logger.debug(f"Starting parallel processing with {self.config.max_workers} workers "
                           f"({'processes' if use_processes else 'threads'})")

                results = list(executor.map(processor_func, data))

                logger.debug(f"Parallel processing completed for {len(data)} items")
                return results

        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}, falling back to sequential processing")
            return [processor_func(item) for item in data]

    def enable_memory_mapping(self, file_path: Path) -> bool:
        """
        Enable memory mapping for large files.

        Args:
            file_path: Path to file

        Returns:
            True if memory mapping was enabled
        """
        if not self.config.enable_memory_mapping:
            return False

        try:
            file_size = file_path.stat().st_size

            # Only use memory mapping for files larger than 10MB
            if file_size > 10 * 1024 * 1024:
                logger.debug(f"Enabling memory mapping for {file_path} ({file_size / 1024 / 1024:.1f}MB)")
                return True

        except Exception as e:
            logger.warning(f"Memory mapping check failed: {e}")

        return False

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized data types
        """
        if not self.config.downcast_dtypes:
            return df

        logger.debug("Optimizing DataFrame data types")
        original_memory = df.memory_usage(deep=True).sum()

        df_optimized = df.copy()

        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['integer']).columns:
            col_series = df_optimized[col]
            if pd.api.types.is_integer_dtype(col_series):
                # Downcast integers
                df_optimized[col] = pd.to_numeric(col_series, downcast='integer')

        for col in df_optimized.select_dtypes(include=['float']).columns:
            col_series = df_optimized[col]
            # Downcast floats
            df_optimized[col] = pd.to_numeric(col_series, downcast='float')

        # Optimize object columns
        for col in df_optimized.select_dtypes(include=['object']).columns:
            col_series = df_optimized[col]
            try:
                # Try to convert to categorical if cardinality is low
                unique_ratio = col_series.nunique() / len(col_series)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df_optimized[col] = col_series.astype('category')
            except:
                pass

        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100

        logger.debug(f"Data type optimization: {original_memory / 1024 / 1024:.1f}MB -> "
                    f"{optimized_memory / 1024 / 1024:.1f}MB ({memory_reduction:.1f}% reduction)")

        return df_optimized

    def process_chunks_optimized(self, file_path: Path, processor_func: Callable,
                                **kwargs) -> pd.DataFrame:
        """
        Process CSV file in optimized chunks.

        Args:
            file_path: Path to CSV file
            processor_func: Function to process each chunk
            **kwargs: Additional parameters for processing

        Returns:
            Combined processed DataFrame
        """
        file_size = file_path.stat().st_size

        # Calculate optimal chunk size
        if 'chunk_size' not in kwargs:
            try:
                import psutil
                available_memory = psutil.virtual_memory().available
                kwargs['chunk_size'] = self.optimize_chunk_size(file_size, available_memory)
            except ImportError:
                kwargs['chunk_size'] = self.config.chunk_size

        logger.info(f"Processing {file_path} in optimized chunks of {kwargs['chunk_size']} rows")

        # Enable memory mapping if appropriate
        use_mmap = self.enable_memory_mapping(file_path)

        # Process chunks
        chunks = []
        chunk_count = 0

        try:
            if use_mmap:
                # Use memory mapping for large files
                chunks = self._process_with_memory_mapping(file_path, processor_func, **kwargs)
            else:
                # Use standard chunked reading
                chunk_reader = pd.read_csv(file_path, chunksize=kwargs['chunk_size'])

                for chunk in chunk_reader:
                    processed_chunk = processor_func(chunk, **kwargs)
                    if processed_chunk is not None and len(processed_chunk) > 0:
                        chunks.append(processed_chunk)

                    chunk_count += 1

                    # Periodic memory cleanup
                    if chunk_count % 10 == 0:
                        self._cleanup_memory()

        except Exception as e:
            logger.error(f"Chunked processing failed: {e}")
            raise

        if not chunks:
            logger.warning("No chunks were successfully processed")
            return pd.DataFrame()

        # Combine results
        logger.info(f"Combining {len(chunks)} processed chunks...")
        result = pd.concat(chunks, ignore_index=False)

        # Optimize final result
        result = self.optimize_dtypes(result)

        return result

    def _process_with_memory_mapping(self, file_path: Path, processor_func: Callable,
                                   **kwargs) -> list:
        """Process file using memory mapping."""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Read header first
                    header_line = mm.readline().decode('utf-8').strip()
                    delimiter = kwargs.get('delimiter', ',')

                    # Find line boundaries for chunking
                    chunk_size_bytes = kwargs['chunk_size'] * 100  # Rough estimate
                    position = len(header_line) + 1

                    while position < mm.size():
                        start_pos = position
                        end_pos = min(position + chunk_size_bytes, mm.size())

                        # Find end of line to avoid splitting rows
                        if end_pos < mm.size():
                            mm.seek(end_pos)
                            while mm.read(1) != b'\n' and mm.tell() < mm.size():
                                end_pos += 1

                        # Extract chunk
                        mm.seek(start_pos)
                        chunk_data = mm.read(end_pos - start_pos).decode('utf-8')

                        if chunk_data.strip():
                            # Convert to DataFrame
                            from io import StringIO
                            chunk_df = pd.read_csv(StringIO(header_line + '\n' + chunk_data))

                            # Process chunk
                            processed_chunk = processor_func(chunk_df, **kwargs)
                            if processed_chunk is not None and len(processed_chunk) > 0:
                                chunks.append(processed_chunk)

                        position = end_pos + 1

        except Exception as e:
            logger.warning(f"Memory mapping processing failed: {e}, falling back to standard processing")
            # Fallback to standard processing
            chunk_reader = pd.read_csv(file_path, chunksize=kwargs['chunk_size'])
            for chunk in chunk_reader:
                processed_chunk = processor_func(chunk, **kwargs)
                if processed_chunk is not None and len(processed_chunk) > 0:
                    chunks.append(processed_chunk)

        return chunks

    def vectorized_operations(self, df: pd.DataFrame, operations: Dict[str, Callable]) -> pd.DataFrame:
        """
        Apply vectorized operations to DataFrame for performance.

        Args:
            df: Input DataFrame
            operations: Dictionary of column -> operation mappings

        Returns:
            DataFrame with vectorized operations applied
        """
        logger.debug(f"Applying {len(operations)} vectorized operations")

        result = df.copy()

        for col_name, operation in operations.items():
            try:
                # Use numpy vectorization when possible
                if hasattr(operation, '__array_ufunc__') or callable(operation):
                    if col_name in result.columns:
                        result[col_name] = operation(result[col_name].values)
                    else:
                        # Create new column
                        result[col_name] = operation(result.values)
            except Exception as e:
                logger.warning(f"Vectorized operation failed for {col_name}: {e}")
                # Fallback to standard pandas operation
                try:
                    result[col_name] = result.apply(operation, axis=1)
                except:
                    logger.error(f"Failed to apply operation {col_name}")

        return result

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        gc.collect()

        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024 * 1024)
            self.memory_usage_history.append(memory_usage)

            # Keep only recent history
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-50:]

        except ImportError:
            pass

    def cache_result(self, key: str, result: Any) -> None:
        """Cache a processing result."""
        if not self.config.cache_results:
            return

        # Simple LRU-like cache
        if len(self.result_cache) >= 100:  # Limit cache size
            # Remove oldest entries
            keys_to_remove = list(self.result_cache.keys())[:-50]
            for k in keys_to_remove:
                del self.result_cache[k]

        self.result_cache[key] = result

    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available."""
        return self.result_cache.get(key)

    def measure_performance(self, operation_func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """
        Measure performance of an operation.

        Args:
            operation_func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, performance_metrics)
        """
        import time
        start_time = time.time()

        # Get initial memory usage
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            initial_memory = 0

        try:
            # Execute operation
            result = operation_func(*args, **kwargs)

            # Calculate metrics
            end_time = time.time()
            processing_time = end_time - start_time

            try:
                final_memory = process.memory_info().rss / (1024 * 1024)
                memory_usage = final_memory - initial_memory
            except:
                memory_usage = 0

            # Estimate rows processed (if result is DataFrame)
            rows_processed = len(result) if hasattr(result, '__len__') else 0
            rows_per_second = rows_processed / processing_time if processing_time > 0 else 0

            metrics = PerformanceMetrics(
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                rows_processed=rows_processed,
                rows_per_second=rows_per_second,
                cache_hit_rate=0.0,  # TODO: Implement cache hit tracking
                parallel_efficiency=0.0  # TODO: Implement parallel efficiency calculation
            )

            logger.debug(f"Performance: {rows_processed} rows in {processing_time:.3f}s "
                        f"({rows_per_second:.0f} rows/sec)")

            return result, metrics

        except Exception as e:
            logger.error(f"Performance measurement failed: {e}")
            raise

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get performance optimization recommendations."""
        recommendations = {}

        # Memory usage analysis
        if self.memory_usage_history:
            avg_memory = np.mean(self.memory_usage_history)
            max_memory = np.max(self.memory_usage_history)

            if avg_memory > self.config.memory_limit_mb * 0.8:
                recommendations['memory'] = [
                    "Reduce memory usage by enabling data type optimization",
                    "Use smaller chunk sizes",
                    "Enable memory mapping for large files"
                ]

            if max_memory > avg_memory * 1.5:
                recommendations['memory_variance'] = [
                    "Memory usage is variable, consider more consistent chunking"
                ]

        # Processing recommendations
        if not self.config.enable_parallel_processing:
            recommendations['parallel'] = [
                "Enable parallel processing for better performance",
                f"Consider using {self.config.max_workers} workers"
            ]

        if not self.config.downcast_dtypes:
            recommendations['data_types'] = [
                "Enable data type downcasting for memory efficiency"
            ]

        if not self.config.enable_memory_mapping:
            recommendations['memory_mapping'] = [
                "Enable memory mapping for large files (>10MB)"
            ]

        return recommendations

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        return {
            'config': {
                'parallel_processing': self.config.enable_parallel_processing,
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size,
                'memory_limit_mb': self.config.memory_limit_mb,
                'memory_mapping': self.config.enable_memory_mapping,
                'downcast_dtypes': self.config.downcast_dtypes
            },
            'memory_usage': {
                'current_mb': self.memory_usage_history[-1] if self.memory_usage_history else 0,
                'average_mb': np.mean(self.memory_usage_history) if self.memory_usage_history else 0,
                'peak_mb': np.max(self.memory_usage_history) if self.memory_usage_history else 0
            },
            'cache': {
                'cached_items': len(self.result_cache),
                'cache_enabled': self.config.cache_results
            }
        }