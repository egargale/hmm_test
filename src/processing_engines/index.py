"""
Processing Engines Index Module

This module provides the main entry point for the processing_engines package,
implementing the ProcessingEngineFactory for dynamic engine selection and processing.
"""

from typing import Optional, Dict, Any, Union
import pandas as pd

from .factory import ProcessingEngineFactory, get_processing_engine_factory, process_with_engine
from .streaming_engine import process_streaming
from .dask_engine import process_dask
from .daft_engine import process_daft

# Re-export the main factory functions and classes
__all__ = [
    'ProcessingEngineFactory',
    'get_processing_engine_factory',
    'process_with_engine',
    'process_streaming',
    'process_dask',
    'process_daft',
]

# Version information
__version__ = "1.0.0"