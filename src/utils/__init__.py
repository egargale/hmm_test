"""
HMM Futures Analysis - Utilities Module

This module provides core utilities for the HMM futures analysis project,
including data types, configuration management, and logging configuration.
"""

from .config import BacktestConfig as ConfigBacktestConfig
from .config import (
    Config,
    HMMConfig,
    LoggingConfig,
    ProcessingConfig,
    create_default_config,
    load_config,
    save_config,
)
from .data_types import (
    BacktestConfig,
    BacktestResult,
    FeatureMatrix,
    FuturesData,
    HMMState,
    PerformanceMetrics,
    PriceData,
    ProbabilityMatrix,
    StateSequence,
    Trade,
)
from .logging_config import (
    get_logger,
    initialize_default_logging,
    log_system_info,
    setup_logging,
    setup_logging_from_config,
)

__all__ = [
    # Data types
    "FuturesData",
    "HMMState",
    "BacktestResult",
    "PerformanceMetrics",
    "BacktestConfig",
    "Trade",
    "PriceData",
    "FeatureMatrix",
    "StateSequence",
    "ProbabilityMatrix",

    # Configuration
    "HMMConfig",
    "ProcessingConfig",
    "ConfigBacktestConfig",
    "LoggingConfig",
    "Config",
    "load_config",
    "save_config",
    "create_default_config",

    # Logging
    "setup_logging",
    "setup_logging_from_config",
    "get_logger",
    "log_system_info",
    "initialize_default_logging"
]
