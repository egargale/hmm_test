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
    f"{PACKAGE}.regime.hmm_adapter",
    f"{PACKAGE}.regime.markov_chain",
    f"{PACKAGE}.data_processing.feature_engineering",
    f"{PACKAGE}.data_processing.messina_features",
    f"{PACKAGE}.data_processing.technical_indicators",
    f"{PACKAGE}.data_processing.csv_auto_detect",
    f"{PACKAGE}.data_processing.csv_parser",
    f"{PACKAGE}.data_processing.csv_format_detector",
    f"{PACKAGE}.data_processing.data_validation",
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

def test_backtesting_init_no_dead_reexports():
    """backtesting/__init__.py must not re-export deleted modules."""
    import hmm_futures_analysis.backtesting as bt

    for attr in ("analyze_performance", "detect_lookahead_bias",
                 "validate_backtest_realism", "calculate_transaction_costs",
                 "validate_backtest_inputs"):
        assert not hasattr(bt, attr), f"Dead re-export still present: backtesting.{attr}"


def test_utils_init_no_dead_reexports():
    """utils/__init__.py must not re-export deleted config or dead types."""
    import hmm_futures_analysis.utils as utils

    for attr in ("Config", "HMMConfig", "ProcessingConfig", "LoggingConfig",
                 "load_config", "save_config", "create_default_config",
                 "ConfigBacktestConfig"):
        assert not hasattr(utils, attr), f"Dead re-export still present: utils.{attr}"
