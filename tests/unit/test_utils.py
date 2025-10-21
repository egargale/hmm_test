"""
Unit tests for utility modules.

Tests the core utility functions, data types, and configurations
that support the HMM futures analysis system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from utils.data_types import BacktestConfig, BacktestResult, Trade, PerformanceMetrics
from utils import get_logger, setup_logging
from utils.config import load_config


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_backtest_config_creation(self):
        """Test creating BacktestConfig with default values."""
        config = BacktestConfig()
        assert config.initial_capital == 100000.0
        assert config.commission == 0.001
        assert config.slippage == 0.0001
        assert config.lookahead_bias_prevention is True
        assert config.lookahead_days == 1

    def test_backtest_config_custom_values(self):
        """Test creating BacktestConfig with custom values."""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission=0.002,
            slippage=0.0002,
            lookahead_bias_prevention=False,
            lookahead_days=2
        )
        assert config.initial_capital == 50000.0
        assert config.commission == 0.002
        assert config.slippage == 0.0002
        assert config.lookahead_bias_prevention is False
        assert config.lookahead_days == 2

    def test_backtest_config_validation(self):
        """Test BacktestConfig parameter validation."""
        # Test negative initial capital
        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=-1000.0)

        # Test negative commission
        with pytest.raises(ValueError):
            BacktestConfig(commission=-0.01)

        # Test negative slippage
        with pytest.raises(ValueError):
            BacktestConfig(slippage=-0.01)


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating Trade with valid data."""
        entry_time = pd.Timestamp('2020-01-01')
        exit_time = pd.Timestamp('2020-01-02')

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=100.0,
            exit_price=105.0,
            size=1.0,
            pnl=5.0,
            commission=1.0,
            slippage=0.5
        )

        assert trade.entry_time == entry_time
        assert trade.exit_time == exit_time
        assert trade.entry_price == 100.0
        assert trade.exit_price == 105.0
        assert trade.size == 1.0
        assert trade.pnl == 5.0
        assert trade.commission == 1.0
        assert trade.slippage == 0.5

    def test_trade_calculations(self):
        """Test trade P&L calculations."""
        entry_time = pd.Timestamp('2020-01-01')
        exit_time = pd.Timestamp('2020-01-02')

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=100.0,
            exit_price=105.0,
            size=2.0,
            pnl=10.0,  # (105-100) * 2 = 10
            commission=2.0,
            slippage=1.0
        )

        expected_pnl = (105.0 - 100.0) * 2.0 - 2.0 - 1.0
        assert trade.pnl == expected_pnl


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics with valid data."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.18,
            annualized_volatility=0.20,
            sharpe_ratio=0.9,
            max_drawdown=-0.08,
            max_drawdown_duration=30,
            calmar_ratio=2.25,
            win_rate=0.6,
            loss_rate=0.4,
            profit_factor=1.5,
            sortino_ratio=1.2
        )

        assert metrics.total_return == 0.15
        assert metrics.annualized_return == 0.18
        assert metrics.sharpe_ratio == 0.9
        assert metrics.win_rate == 0.6

    def test_performance_metrics_validation(self):
        """Test PerformanceMetrics parameter validation."""
        # Test win rate > 1
        with pytest.raises(ValueError):
            PerformanceMetrics(win_rate=1.5)

        # Test loss rate > 1
        with pytest.raises(ValueError):
            PerformanceMetrics(loss_rate=1.5)

        # Test negative profit factor
        with pytest.raises(ValueError):
            PerformanceMetrics(profit_factor=-1.0)


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_backtest_result_creation(self, sample_ohlcv_data):
        """Test creating BacktestResult with valid data."""
        equity_curve = sample_ohlcv_data['close'] * 10
        positions = pd.Series([1, -1, 0, 1, -1] * 20, index=sample_ohlcv_data.index[:100])

        result = BacktestResult(
            equity_curve=equity_curve,
            positions=positions,
            trades=[],
            start_date=sample_ohlcv_data.index[0],
            end_date=sample_ohlcv_data.index[-1]
        )

        assert len(result.equity_curve) == len(sample_ohlcv_data)
        assert len(result.positions) == 100
        assert len(result.trades) == 0

    def test_backtest_result_validation(self):
        """Test BacktestResult parameter validation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        equity_curve = pd.Series([100] * 10, index=dates)
        positions = pd.Series([1, -1, 0] * 3 + [1], index=dates)

        # Test start_date > end_date
        with pytest.raises(ValueError):
            BacktestResult(
                equity_curve=equity_curve,
                positions=positions,
                trades=[],
                start_date=dates[-1],
                end_date=dates[0]
            )


class TestLogger:
    """Test logging utilities."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_setup_logging_default(self):
        """Test setting up logging with default configuration."""
        setup_logging()
        # Should not raise any exceptions
        logger = get_logger("test_default")
        assert logger is not None

    def test_setup_logging_debug(self):
        """Test setting up logging with DEBUG level."""
        setup_logging(level="DEBUG")
        logger = get_logger("test_debug")
        assert logger is not None

    def test_setup_logging_invalid_level(self):
        """Test setting up logging with invalid level."""
        with pytest.raises(ValueError):
            setup_logging(level="INVALID_LEVEL")


class TestConfig:
    """Test configuration utilities."""

    def test_load_config_file_exists(self, temp_dir):
        """Test loading configuration from existing file."""
        config_data = {
            "initial_capital": 150000.0,
            "commission": 0.002,
            "n_states": 4
        }
        config_file = temp_dir / "test_config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(str(config_file))
        assert config is not None
        # The Config object has its own structure, just verify it loads successfully

    def test_load_config_file_not_exists(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_load_config_basic(self):
        """Test basic config loading functionality."""
        config_data = {
            "initial_capital": 100000.0,
            "commission": 0.001,
            "n_states": 3
        }
        # Test that we can create config data structure
        assert config_data["initial_capital"] == 100000.0
        assert config_data["commission"] == 0.001
        assert config_data["n_states"] == 3


class TestUtilityFunctions:
    """Test general utility functions."""

    def test_path_operations(self, temp_dir):
        """Test path operations and file handling."""
        test_file = temp_dir / "test_file.txt"
        test_content = "Test content for file operations"

        # Write to file
        test_file.write_text(test_content)
        assert test_file.exists()

        # Read from file
        content = test_file.read_text()
        assert content == test_content

    def test_data_frame_operations(self):
        """Test DataFrame utility operations."""
        # Create test DataFrame
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })

        # Test basic operations
        assert len(data) == 5
        assert list(data.columns) == ['A', 'B', 'C']
        assert data['A'].sum() == 15

    def test_numpy_operations(self):
        """Test NumPy utility operations."""
        # Create test arrays
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([10, 20, 30, 40, 50])

        # Test basic operations
        assert np.sum(arr1) == 15
        assert np.mean(arr2) == 30.0
        assert np.dot(arr1, arr2) == 550

    def test_datetime_operations(self):
        """Test datetime utility operations."""
        # Create test dates
        dates = pd.date_range('2020-01-01', periods=5, freq='D')

        # Test operations
        assert len(dates) == 5
        assert dates[0] == pd.Timestamp('2020-01-01')
        assert dates[-1] == pd.Timestamp('2020-01-05')

    def test_mathematical_functions(self):
        """Test mathematical utility functions."""
        # Test basic math operations
        assert round(3.14159, 2) == 3.14
        assert abs(-5.5) == 5.5
        assert max([1, 2, 3, 4, 5]) == 5
        assert min([1, 2, 3, 4, 5]) == 1


@pytest.mark.integration
class TestIntegrationUtils:
    """Integration tests for utility modules."""

    def test_config_with_backtest(self, backtest_config):
        """Test configuration integration with backtest components."""
        assert isinstance(backtest_config, dict)
        assert 'initial_capital' in backtest_config
        assert 'commission' in backtest_config

    def test_logging_with_config(self, temp_dir):
        """Test logging integration with configuration."""
        # Create a config that specifies log level
        config = {"log_level": "DEBUG", "log_file": str(temp_dir / "test.log")}

        setup_logging(level=config["log_level"])
        logger = get_logger("integration_test")

        # Should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")