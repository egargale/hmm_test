"""
Processing Engine Factory Module

Implements a factory pattern for selecting and using different processing engines
(Pandas Streaming, Dask, Daft) based on data size, memory constraints, and user preferences.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union

import pandas as pd

from .streaming_engine import process_streaming, estimate_processing_time
from .dask_engine import process_dask, compute_with_progress, benchmark_dask_engines
from .daft_engine import process_daft, compute_daft_with_progress, benchmark_daft_engine
from utils import get_logger, ProcessingConfig

logger = get_logger(__name__)


class ProcessingEngineFactory:
    """
    Factory class for creating and managing different processing engines.

    Automatically selects the best engine based on data size, available resources,
    and user preferences.
    """

    def __init__(self):
        self._engine_status = self._check_engine_availability()
        self._benchmark_cache = {}

    def _check_engine_availability(self) -> Dict[str, bool]:
        """Check which processing engines are available."""
        status = {
            'streaming': True,  # Always available (pandas)
        }

        # Check Dask availability
        try:
            import dask
            import dask.dataframe as dd
            status['dask'] = True
        except ImportError:
            status['dask'] = False
            logger.debug("Dask not available")

        # Check Daft availability
        try:
            import daft
            status['daft'] = True
        except ImportError:
            status['daft'] = False
            logger.debug("Daft not available")

        return status

    def get_available_engines(self) -> list:
        """Get list of available processing engines."""
        return [name for name, available in self._engine_status.items() if available]

    def recommend_engine(
        self,
        csv_path: str,
        memory_limit_gb: float = 8.0,
        prefer_speed: bool = True,
        prefer_memory_efficiency: bool = False,
        require_distributed: bool = False
    ) -> str:
        """
        Recommend the best processing engine based on file characteristics and requirements.

        Args:
            csv_path: Path to the CSV file
            memory_limit_gb: Available memory limit in GB
            prefer_speed: Whether to prioritize speed over memory efficiency
            prefer_memory_efficiency: Whether to prioritize memory efficiency
            require_distributed: Whether distributed processing is required

        Returns:
            str: Recommended engine name ('streaming', 'dask', 'daft')
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        available_engines = self.get_available_engines()

        logger.info(f"Recommending engine for {file_size_mb:.1f}MB file")
        logger.info(f"Available engines: {available_engines}")
        logger.info(f"Requirements: speed={prefer_speed}, memory_efficiency={prefer_memory_efficiency}, distributed={require_distributed}")

        # Decision matrix
        if require_distributed:
            # Must use distributed engine
            if 'dask' in available_engines:
                recommendation = 'dask'
            elif 'daft' in available_engines:
                recommendation = 'daft'
            else:
                logger.warning("Distributed processing required but no distributed engines available")
                recommendation = 'streaming'

        elif file_size_mb < 100:  # Small files (< 100MB)
            recommendation = 'streaming'
            logger.info("Small file detected - recommending streaming engine")

        elif file_size_mb < 1000:  # Medium files (100MB - 1GB)
            if prefer_speed and 'dask' in available_engines:
                recommendation = 'dask'
            elif prefer_memory_efficiency or 'dask' not in available_engines:
                recommendation = 'streaming'
            elif 'daft' in available_engines:
                recommendation = 'daft'
            else:
                recommendation = 'streaming'
            logger.info("Medium file detected - engine choice based on preferences")

        else:  # Large files (> 1GB)
            if prefer_memory_efficiency:
                recommendation = 'streaming'
            elif 'dask' in available_engines:
                recommendation = 'dask'
            elif 'daft' in available_engines:
                recommendation = 'daft'
            else:
                recommendation = 'streaming'
            logger.info("Large file detected - prioritizing distributed/memory-efficient engines")

        logger.info(f"Recommended engine: {recommendation}")
        return recommendation

    def create_processor(
        self,
        engine: str,
        csv_path: str,
        config: ProcessingConfig,
        **engine_kwargs
    ) -> Union[pd.DataFrame, "dd.DataFrame", "daft.DataFrame"]:
        """
        Create a processor using the specified engine.

        Args:
            engine: Engine name ('streaming', 'dask', 'daft')
            csv_path: Path to CSV file
            config: Processing configuration
            **engine_kwargs: Engine-specific parameters

        Returns:
            Processed DataFrame (type depends on engine)
        """
        if engine not in self.get_available_engines():
            raise ValueError(f"Engine '{engine}' is not available. Available: {self.get_available_engines()}")

        logger.info(f"Creating processor with {engine} engine")

        if engine == 'streaming':
            return self._create_streaming_processor(csv_path, config, **engine_kwargs)
        elif engine == 'dask':
            return self._create_dask_processor(csv_path, config, **engine_kwargs)
        elif engine == 'daft':
            return self._create_daft_processor(csv_path, config, **engine_kwargs)
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _create_streaming_processor(
        self,
        csv_path: str,
        config: ProcessingConfig,
        chunk_size: Optional[int] = None,
        memory_limit_gb: float = 8.0,
        show_progress: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Create streaming processor."""
        return process_streaming(
            csv_path=csv_path,
            config=config,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            show_progress=show_progress
        )

    def _create_dask_processor(
        self,
        csv_path: str,
        config: ProcessingConfig,
        scheduler: str = "threads",
        npartitions: Optional[int] = None,
        memory_limit: str = "2GB",
        show_progress: bool = True,
        **kwargs
    ) -> "dd.DataFrame":
        """Create Dask processor."""
        return process_dask(
            csv_path=csv_path,
            config=config,
            scheduler=scheduler,
            npartitions=npartitions,
            memory_limit=memory_limit,
            show_progress=show_progress
        )

    def _create_daft_processor(
        self,
        csv_path: str,
        config: ProcessingConfig,
        npartitions: Optional[int] = None,
        memory_limit: str = "2GB",
        show_progress: bool = True,
        use_accelerators: bool = True,
        **kwargs
    ) -> "daft.DataFrame":
        """Create Daft processor."""
        return process_daft(
            csv_path=csv_path,
            config=config,
            npartitions=npartitions,
            memory_limit=memory_limit,
            show_progress=show_progress,
            use_accelerators=use_accelerators
        )

    def compute_result(
        self,
        processed_data: Union[pd.DataFrame, "dd.DataFrame", "daft.DataFrame"],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Compute the final result from processed data.

        Args:
            processed_data: Processed data from any engine
            show_progress: Whether to show progress

        Returns:
            pd.DataFrame: Final computed result
        """
        # Check if it's already a pandas DataFrame
        if isinstance(processed_data, pd.DataFrame):
            logger.info("Data already computed as pandas DataFrame")
            return processed_data

        # Check if it's a Dask DataFrame
        try:
            import dask.dataframe as dd
            if isinstance(processed_data, dd.DataFrame):
                logger.info("Computing Dask DataFrame")
                return compute_with_progress(processed_data, show_progress=show_progress)
        except ImportError:
            pass

        # Check if it's a Daft DataFrame
        try:
            import daft
            if hasattr(processed_data, 'to_pandas'):  # Daft DataFrame
                logger.info("Computing Daft DataFrame")
                return compute_daft_with_progress(processed_data, show_progress=show_progress)
        except ImportError:
            pass

        # Fallback: try to convert to pandas
        logger.warning("Unknown data type, attempting to convert to pandas")
        try:
            if hasattr(processed_data, 'to_pandas'):
                return processed_data.to_pandas()
            elif hasattr(processed_data, 'compute'):
                return processed_data.compute()
            else:
                return pd.DataFrame(processed_data)
        except Exception as e:
            logger.error(f"Failed to compute result: {e}")
            raise

    def benchmark_engines(
        self,
        csv_path: str,
        engines: Optional[list] = None,
        cache_results: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark all available engines for performance comparison.

        Args:
            csv_path: Path to test CSV file
            engines: List of engines to test (None = all available)
            cache_results: Whether to cache results

        Returns:
            Dict with benchmark results for each engine
        """
        if engines is None:
            engines = self.get_available_engines()

        # Check cache first
        cache_key = f"{csv_path}_{sorted(engines)}"
        if cache_results and cache_key in self._benchmark_cache:
            logger.info("Using cached benchmark results")
            return self._benchmark_cache[cache_key]

        logger.info(f"Benchmarking engines: {engines}")
        results = {}

        # Create test configuration
        config = ProcessingConfig(
            engine_type="auto",  # Will be overridden
            chunk_size=1000
        )

        for engine in engines:
            logger.info(f"Benchmarking {engine} engine...")

            try:
                start_time = time.time()

                if engine == 'streaming':
                    # Estimate processing time first
                    estimate = estimate_processing_time(csv_path)
                    logger.info(f"Estimated streaming processing time: {estimate.get('estimated_total_time', 0):.2f}s")

                    # Actually process
                    result = self.create_processor(engine, csv_path, config, show_progress=False)

                    end_time = time.time()
                    processing_time = end_time - start_time

                    results[engine] = {
                        'success': True,
                        'time': processing_time,
                        'rows': len(result),
                        'columns': len(result.columns),
                        'memory_mb': result.memory_usage(deep=True).sum() / (1024 * 1024),
                        'estimated_time': estimate.get('estimated_total_time', 0)
                    }

                elif engine == 'dask':
                    # Use Dask benchmark function
                    dask_results = benchmark_dask_engines(csv_path, schedulers=["threads"])

                    if 'threads' in dask_results and 1000 in dask_results['threads']:
                        dask_result = dask_results['threads'][1000]
                        results[engine] = {
                            'success': True,
                            'time': dask_result['time'],
                            'rows': dask_result['rows'],
                            'columns': dask_result['columns'],
                            'memory_mb': dask_result['memory_mb'],
                            'partitions': 4  # Default
                        }
                    else:
                        results[engine] = {'success': False, 'error': 'Dask benchmark failed'}

                elif engine == 'daft':
                    # Use Daft benchmark function
                    daft_results = benchmark_daft_engine(csv_path, partition_counts=[1])

                    if 'partitions_1' in daft_results and daft_results['partitions_1'].get('success'):
                        daft_result = daft_results['partitions_1']
                        results[engine] = {
                            'success': True,
                            'time': daft_result['time'],
                            'rows': daft_result['rows'],
                            'columns': daft_result['columns'],
                            'memory_mb': daft_result['memory_mb'],
                            'partitions': 1
                        }
                    else:
                        results[engine] = {'success': False, 'error': 'Daft benchmark failed'}

                logger.info(f"  {engine}: {results[engine].get('time', 0):.2f}s, "
                           f"{results[engine].get('rows', 0)} rows")

            except Exception as e:
                logger.error(f"  {engine}: FAILED - {e}")
                results[engine] = {'success': False, 'error': str(e)}

        # Cache results
        if cache_results:
            self._benchmark_cache[cache_key] = results

        return results

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about all available engines."""
        info = {
            'available_engines': self.get_available_engines(),
            'engine_status': self._engine_status,
            'engine_details': {}
        }

        # Streaming engine info
        info['engine_details']['streaming'] = {
            'type': 'pandas',
            'distributed': False,
            'memory_efficient': True,
            'best_for': ['small_files', 'medium_files', 'memory_constrained']
        }

        # Dask engine info
        if self._engine_status.get('dask'):
            try:
                from dask_engine import get_dask_cluster_info
                cluster_info = get_dask_cluster_info()
                info['engine_details']['dask'] = {
                    'type': 'dask',
                    'distributed': True,
                    'memory_efficient': True,
                    'best_for': ['large_files', 'distributed_processing', 'parallel_computation'],
                    'cluster_info': cluster_info
                }
            except:
                info['engine_details']['dask'] = {
                    'type': 'dask',
                    'distributed': True,
                    'memory_efficient': True,
                    'best_for': ['large_files', 'distributed_processing'],
                    'cluster_info': 'Failed to retrieve'
                }

        # Daft engine info
        if self._engine_status.get('daft'):
            try:
                from daft_engine import get_daft_cluster_info
                daft_info = get_daft_cluster_info()
                info['engine_details']['daft'] = {
                    'type': 'daft',
                    'distributed': True,
                    'memory_efficient': True,
                    'best_for': ['large_files', 'cloud_native', 'gpu_acceleration'],
                    'cluster_info': daft_info
                }
            except:
                info['engine_details']['daft'] = {
                    'type': 'daft',
                    'distributed': True,
                    'memory_efficient': True,
                    'best_for': ['large_files', 'cloud_native'],
                    'cluster_info': 'Failed to retrieve'
                }

        return info


# Global factory instance
_factory_instance = None


def get_processing_engine_factory() -> ProcessingEngineFactory:
    """Get the global processing engine factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = ProcessingEngineFactory()
    return _factory_instance


def process_with_engine(
    csv_path: str,
    config: ProcessingConfig,
    engine: Optional[str] = None,
    compute_result: bool = True,
    show_progress: bool = True,
    **engine_kwargs
) -> pd.DataFrame:
    """
    Convenience function to process data with automatic or specified engine selection.

    Args:
        csv_path: Path to CSV file
        config: Processing configuration
        engine: Specific engine to use (None = auto-select)
        compute_result: Whether to compute the final result
        show_progress: Whether to show progress
        **engine_kwargs: Engine-specific parameters

    Returns:
        pd.DataFrame: Processed data
    """
    factory = get_processing_engine_factory()

    # Auto-select engine if not specified
    if engine is None:
        engine = factory.recommend_engine(
            csv_path=csv_path,
            memory_limit_gb=engine_kwargs.get('memory_limit_gb', 8.0),
            prefer_speed=engine_kwargs.get('prefer_speed', True),
            prefer_memory_efficiency=engine_kwargs.get('prefer_memory_efficiency', False)
        )

    logger.info(f"Processing with {engine} engine")

    # Create processor
    processed_data = factory.create_processor(engine, csv_path, config, **engine_kwargs)

    # Compute result if requested
    if compute_result:
        result = factory.compute_result(processed_data, show_progress=show_progress)
        return result
    else:
        return processed_data