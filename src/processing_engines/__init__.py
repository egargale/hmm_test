"""
Processing Engines Module

This module provides multiple processing engines for efficient and scalable
data processing, supporting streaming (Pandas), Dask, and Daft for different
dataset sizes and memory constraints.

The main entry point is through the ProcessingEngineFactory, which can automatically
select the best engine based on data size and system resources.
"""

# Import the main factory and engine functions
try:
    from .index import (
        ProcessingEngineFactory,
        get_processing_engine_factory,
        process_with_engine,
        process_streaming,
        process_dask,
        process_daft
    )
    _imports_available = True
except ImportError as e:
    print(f"Warning: Could not import processing engines: {e}")
    _imports_available = False
    ProcessingEngineFactory = None
    get_processing_engine_factory = None
    process_with_engine = None
    process_streaming = None
    process_dask = None
    process_daft = None

__all__ = []

if _imports_available:
    __all__.extend([
        "ProcessingEngineFactory",
        "get_processing_engine_factory",
        "process_with_engine",
        "process_streaming",
        "process_dask",
        "process_daft"
    ])

# Convenience function for quick engine selection
def get_engine(engine_type: str):
    """
    Get a specific processing engine function.

    Args:
        engine_type: Type of engine ('streaming', 'dask', 'daft')

    Returns:
        Processing function for the specified engine

    Raises:
        ValueError: If engine type is not supported
    """
    if engine_type == "streaming":
        return process_streaming
    elif engine_type == "dask":
        return process_dask
    elif engine_type == "daft":
        return process_daft
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")

if _imports_available:
    __all__.append("get_engine")