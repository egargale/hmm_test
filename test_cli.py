#!/usr/bin/env python3
"""
Integration tests for the HMM futures program CLI.
"""

import os
import tempfile
import unittest
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

class TestHMMFuturesCLI(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'DateTime': pd.date_range('2023-01-01', periods=50, freq='h'),
            'Open': np.linspace(100, 149, 50),
            'High': np.linspace(101, 150, 50),
            'Low': np.linspace(99, 148, 50),
            'Close': np.linspace(100.5, 149.5, 50),
            'Volume': np.linspace(1000, 1490, 50)
        })
        
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv = Path(self.temp_dir.name) / 'test_data.csv'
        self.test_data.to_csv(self.temp_csv, index=False)
        
        # Activate virtual environment
        self.env = os.environ.copy()
        self.env['PATH'] = f"{Path(self.temp_dir.name).parent.parent}/.venv/bin:" + self.env['PATH']
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def run_command(self, cmd):
        """Run a command and return the result."""
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent,
            env=self.env
        )
        return result
    
    def test_basic_execution(self):
        """Test basic execution of the program."""
        cmd = f"python main.py {self.temp_csv} --n_states 2 --max_iter 5"
        result = self.run_command(cmd)
        
        # Check that the command executed successfully
        self.assertEqual(result.returncode, 0, f"Command failed with output: {result.stderr}")
        
        # Check that output files were created
        output_csv = self.temp_csv.with_suffix('.hmm_states.csv')
        self.assertTrue(output_csv.exists(), "Output CSV file was not created")
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        model_file = Path(self.temp_dir.name) / 'test_model.pkl'
        
        # Train and save model
        cmd1 = f"python main.py {self.temp_csv} --n_states 2 --max_iter 5 --model-out {model_file}"
        result1 = self.run_command(cmd1)
        self.assertEqual(result1.returncode, 0, f"Model saving failed with output: {result1.stderr}")
        self.assertTrue(model_file.exists(), "Model file was not created")
        
        # Load and use model
        cmd2 = f"python main.py {self.temp_csv} --model-path {model_file}"
        result2 = self.run_command(cmd2)
        self.assertEqual(result2.returncode, 0, f"Model loading failed with output: {result2.stderr}")
    
    def test_backtesting(self):
        """Test backtesting functionality."""
        cmd = f"python main.py {self.temp_csv} --n_states 2 --max_iter 5 --backtest"
        result = self.run_command(cmd)
        
        # Check that the command executed successfully
        self.assertEqual(result.returncode, 0, f"Command failed with output: {result.stderr}")
        
        # Check that backtest output file was created
        backtest_csv = self.temp_csv.with_suffix('.backtest.csv')
        self.assertTrue(backtest_csv.exists(), "Backtest CSV file was not created")
    
    def test_prevent_lookahead(self):
        """Test lookahead bias prevention."""
        cmd = f"python main.py {self.temp_csv} --n_states 2 --max_iter 5 --prevent-lookahead"
        result = self.run_command(cmd)
        
        # Check that the command executed successfully
        self.assertEqual(result.returncode, 0, f"Command failed with output: {result.stderr}")
        
        # Check that output file was created
        output_csv = self.temp_csv.with_suffix('.hmm_states.csv')
        self.assertTrue(output_csv.exists(), "Output CSV file was not created")
    
    def test_chunksize_parameter(self):
        """Test chunk size parameter."""
        cmd = f"python main.py {self.temp_csv} --n_states 2 --max_iter 5 --chunksize 30"
        result = self.run_command(cmd)
        
        # Check that the command executed successfully
        self.assertEqual(result.returncode, 0, f"Command failed with output: {result.stderr}")
        
        # Check that output file was created
        output_csv = self.temp_csv.with_suffix('.hmm_states.csv')
        self.assertTrue(output_csv.exists(), "Output CSV file was not created")

if __name__ == '__main__':
    unittest.main()