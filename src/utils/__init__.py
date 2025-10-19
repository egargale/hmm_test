"""
HMM Futures Analysis - Utilities Module

This module provides core utilities for the HMM futures analysis project,
including data types, configuration management, and logging configuration.
"""

from .data_types import (
    FuturesData,
    HMMState,
    BacktestResult,
    PerformanceMetrics,
    BacktestConfig,
    Trade,
    PriceData,
    FeatureMatrix,
    StateSequence,
    ProbabilityMatrix
)

from .config import (
    HMMConfig,
    ProcessingConfig,
    BacktestConfig as ConfigBacktestConfig,
    LoggingConfig,
    Config,
    load_config,
    save_config,
    create_default_config
)

from .logging_config import (
    setup_logging,
    setup_logging_from_config,
    get_logger,
    log_system_info,
    initialize_default_logging
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