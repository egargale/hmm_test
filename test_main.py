#!/usr/bin/env python3
"""
Unit tests for the HMM futures program.
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import the main module
import sys
sys.path.insert(0, str(Path(__file__).parent))

from main import add_features, stream_features, simple_backtest, perf_metrics

class TestHMMFutures(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'DateTime': pd.date_range('2023-01-01', periods=20, freq='H'),
            'Open': np.linspace(100, 119, 20),
            'High': np.linspace(101, 120, 20),
            'Low': np.linspace(99, 118, 20),
            'Close': np.linspace(100.5, 119.5, 20),
            'Volume': np.linspace(1000, 1190, 20)
        })
        self.test_data.set_index('DateTime', inplace=True)
        
        # Create a temporary CSV file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv = Path(self.temp_dir.name) / 'test_data.csv'
        self.test_data.to_csv(self.temp_csv)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_add_features(self):
        """Test the add_features function."""
        # Apply feature engineering
        result = add_features(self.test_data)
        
        # Check that the result has the expected columns
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_ret', 'atr', 'roc']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that NaN rows have been removed
        self.assertFalse(result.isnull().any().any())
        
        # Check that we have fewer rows than the original (due to NaN removal)
        self.assertLess(len(result), len(self.test_data))
    
    def test_stream_features(self):
        """Test the stream_features function."""
        # Test with default chunk size
        result = stream_features(self.temp_csv)
        
        # Check that the result has the expected columns
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_ret', 'atr', 'roc']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that NaN rows have been removed
        self.assertFalse(result.isnull().any().any())
    
    def test_simple_backtest(self):
        """Test the simple_backtest function."""
        # Apply feature engineering first
        feat_df = add_features(self.test_data)
        
        # Create dummy states (alternating 0 and 2)
        states = np.array([0, 2] * (len(feat_df) // 2) + [0] * (len(feat_df) % 2))
        
        # Run backtest
        result = simple_backtest(feat_df, states)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that we have the expected number of results
        # (should be one less than the input due to shift(-1) and dropna())
        self.assertEqual(len(result), len(feat_df) - 1)
    
    def test_perf_metrics(self):
        """Test the perf_metrics function."""
        # Create a simple equity curve
        equity_curve = pd.Series(np.linspace(0, 1, 100))
        
        # Calculate performance metrics
        sharpe, max_dd = perf_metrics(equity_curve)
        
        # Check that we get numeric results
        self.assertIsInstance(sharpe, (int, float))
        self.assertIsInstance(max_dd, (int, float))

if __name__ == '__main__':
    unittest.main()