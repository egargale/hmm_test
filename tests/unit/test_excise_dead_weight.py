"""Safeguard tests for ADR-001: dead modules must not exist, alive modules must."""

import importlib
import pathlib

import pytest

PACKAGE = "hmm_futures_analysis"
ROOT = pathlib.Path(__file__).resolve().parent.parent / PACKAGE


# -- Dead modules: must NOT exist on disk ----------------------------------

DEAD_PATHS = [
    "hmm_models/base.py",
    "hmm_models/factory.py",
    "hmm_models/gaussian_hmm.py",
    "hmm_models/gmm_hmm.py",
    "hmm_models/__init__.py",
    "model_training/hmm_trainer.py",
    "model_training/inference_engine.py",
    "model_training/model_persistence.py",
    "model_training/__init__.py",
    "backtesting/performance_analyzer.py",
    "backtesting/bias_prevention.py",
    "backtesting/utils.py",
    "utils/config.py",
    "data_processing/csv_parser.py",
    "data_processing/csv_format_detector.py",
    "data_processing/data_validation.py",
]

DEAD_DIRS = [
    "hmm_models",
    "model_training",
]


@pytest.mark.parametrize("relpath", DEAD_PATHS)
def test_dead_file_removed(relpath):
    """Dead module files must not exist."""
    assert not (ROOT / relpath).exists(), f"Dead file still present: {relpath}"


@pytest.mark.parametrize("dirname", DEAD_DIRS)
def test_dead_dir_removed(dirname):
    """Dead module directories must not exist."""
    assert not (ROOT / dirname).exists(), f"Dead directory still present: {dirname}"


@pytest.mark.parametrize("relpath", DEAD_PATHS)
def test_dead_module_not_importable(relpath):
    """Dead modules must not be importable."""
    module_path = relpath.replace("/", ".").replace("\\", ".").removesuffix(".py")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{PACKAGE}.{module_path}")


# -- Alive modules: must still exist and be importable ----------------------

ALIVE_MODULES = [
    f"{PACKAGE}.cli",
    f"{PACKAGE}.regime.pipeline",
    f"{PACKAGE}.regime.walk_forward",
    f"{PACKAGE}.regime.markov_chain",
    f"{PACKAGE}.data_processing.feature_engineering",
    f"{PACKAGE}.data_processing.messina_features",
    f"{PACKAGE}.data_processing.technical_indicators",
    f"{PACKAGE}.data_processing.csv_auto_detect",
    f"{PACKAGE}.backtesting.performance_metrics",
    f"{PACKAGE}.utils.data_types",
    f"{PACKAGE}.utils.logging_config",
]


@pytest.mark.parametrize("modname", ALIVE_MODULES)
def test_alive_module_importable(modname):
    """Alive modules must be importable."""
    mod = importlib.import_module(modname)
    assert mod is not None


# -- Package surface: __init__.py must not re-export dead symbols ----------


DEAD_PERFORMANCE_FUNCTIONS = [
    "calculate_performance",
    "infer_trading_frequency",
    "validate_performance_metrics",
    "create_performance_summary",
]


@pytest.mark.parametrize("func_name", DEAD_PERFORMANCE_FUNCTIONS)
def test_dead_performance_function_removed(func_name):
    """Dead functions must not exist in performance_metrics module."""
    import hmm_futures_analysis.backtesting.performance_metrics as pm

    assert not hasattr(pm, func_name), (
        f"Dead function still present: performance_metrics.{func_name}"
    )


def test_data_processing_init_no_dead_reexports():
    """data_processing/__init__.py must not re-export deleted modules."""
    import hmm_futures_analysis.data_processing as dp

    for attr in (
        "process_csv",
        "validate_data",
        "CSVFormatDetector",
        "CSVFormat",
        "DetectionResult",
        # Issue #55: dead function removed from technical_indicators.py
        "get_available_indicators",
    ):
        assert not hasattr(dp, attr), (
            f"Dead re-export still present: data_processing.{attr}"
        )


def test_backtesting_init_no_dead_reexports():
    """backtesting/__init__.py must not re-export deleted modules."""
    import hmm_futures_analysis.backtesting as bt

    for attr in (
        "analyze_performance",
        "detect_lookahead_bias",
        "validate_backtest_realism",
        "calculate_transaction_costs",
        "validate_backtest_inputs",
        # Issue #55: dead functions removed from performance_metrics.py
        "calculate_performance",
    ):
        assert not hasattr(bt, attr), (
            f"Dead re-export still present: backtesting.{attr}"
        )


DEAD_INDICATOR_FUNCTIONS = [
    "get_available_indicators",
]


@pytest.mark.parametrize("func_name", DEAD_INDICATOR_FUNCTIONS)
def test_dead_indicator_function_removed(func_name):
    """Dead functions must not exist in technical_indicators module."""
    import hmm_futures_analysis.data_processing.technical_indicators as ti

    assert not hasattr(ti, func_name), (
        f"Dead function still present: technical_indicators.{func_name}"
    )


DEAD_DATACLASSES = [
    "FuturesData",
    "HMMState",
    "Trade",
    "BacktestResult",
    "PerformanceMetrics",
    "BacktestConfig",
    "CSVFormat",
    "ProcessingStats",
]


@pytest.mark.parametrize("class_name", DEAD_DATACLASSES)
def test_dead_dataclass_removed(class_name):
    """Dead dataclasses must not exist in data_types module."""
    import hmm_futures_analysis.utils.data_types as dt

    assert not hasattr(dt, class_name), (
        f"Dead dataclass still present: data_types.{class_name}"
    )


ALIVE_TYPE_ALIASES = [
    "PriceData",
    "FeatureMatrix",
    "StateSequence",
    "ProbabilityMatrix",
]


@pytest.mark.parametrize("alias_name", ALIVE_TYPE_ALIASES)
def test_alive_type_alias_exists(alias_name):
    """Type aliases must survive in data_types module."""
    import hmm_futures_analysis.utils.data_types as dt

    assert hasattr(dt, alias_name), f"Type alias missing from data_types: {alias_name}"


def test_utils_init_no_dead_dataclasses():
    """utils/__init__.py must not re-export dead dataclasses."""
    import hmm_futures_analysis.utils as utils

    for attr in (
        "FuturesData",
        "HMMState",
        "Trade",
        "BacktestResult",
        "PerformanceMetrics",
        "BacktestConfig",
        "CSVFormat",
        "ProcessingStats",
    ):
        assert not hasattr(utils, attr), f"Dead re-export still present: utils.{attr}"


def test_utils_init_no_dead_reexports():
    """utils/__init__.py must not re-export deleted config or dead types."""
    import hmm_futures_analysis.utils as utils

    for attr in (
        "Config",
        "HMMConfig",
        "ProcessingConfig",
        "LoggingConfig",
        "load_config",
        "save_config",
        "create_default_config",
        "ConfigBacktestConfig",
    ):
        assert not hasattr(utils, attr), f"Dead re-export still present: utils.{attr}"
