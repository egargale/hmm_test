"""
Processing Engines Module

This module provides multiple processing engines for efficient and scalable
data processing, supporting streaming (Pandas), Dask, and Daft for different
dataset sizes and memory constraints.
"""

try:
    from .streaming_engine import process_streaming
except ImportError:
    process_streaming = None

try:
    from .dask_engine import process_dask
except ImportError:
    process_dask = None

try:
    from .daft_engine import process_daft
except ImportError:
    process_daft = None

try:
    from .factory import ProcessingEngineFactory, process_with_engine, get_processing_engine_factory
except ImportError:
    ProcessingEngineFactory = None
    process_with_engine = None
    get_processing_engine_factory = None

__all__ = []

if process_streaming is not None:
    __all__.append("process_streaming")

if process_dask is not None:
    __all__.append("process_dask")

if process_daft is not None:
    __all__.append("process_daft")

if ProcessingEngineFactory is not None:
    __all__.extend(["ProcessingEngineFactory", "process_with_engine", "get_processing_engine_factory"])