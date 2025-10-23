"""
Simple Test suite for CLI Integration and Orchestration (Task 9)

Tests the core CLI functionality without complex dependencies.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, "src")

from click.testing import CliRunner

try:
    from cli_simple import cli
except ImportError:
    print("CLI module not found, skipping tests")
    cli = None


def create_test_csv_data(n_samples=50, filename="test_data.csv"):
    """Create test CSV data for CLI testing."""
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_samples)
    close_prices = base_price * np.cumprod(1 + returns)

    # Create OHLCV data
    high_spread = np.random.uniform(0.005, 0.02, n_samples)
    low_spread = np.random.uniform(0.005, 0.02, n_samples)

    data = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices * (1 + high_spread),
            "low": close_prices * (1 - low_spread),
            "close": close_prices,
            "volume": np.random.uniform(1000, 5000, n_samples),
        },
        index=dates,
    )

    data.to_csv(filename)
    return filename


def test_cli_import():
    """Test that CLI module can be imported."""
    try:
        from cli_simple import cli

        return True
    except ImportError as e:
        print(f"Failed to import CLI: {e}")
        return False


def test_cli_help():
    """Test CLI help functionality."""
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        if result.exit_code == 0 and "HMM Futures Analysis CLI" in result.output:
            print("✓ CLI help test passed")
            return True
        else:
            print(f"✗ CLI help test failed: {result.output}")
            return False
    except Exception as e:
        print(f"✗ CLI help test failed with exception: {e}")
        return False


def test_cli_version():
    """Test CLI version command."""
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])

        if result.exit_code == 0 and "HMM Futures Analysis CLI v1.0.0" in result.output:
            print("✓ CLI version test passed")
            return True
        else:
            print(f"✗ CLI version test failed: {result.output}")
            return False
    except Exception as e:
        print(f"✗ CLI version test failed with exception: {e}")
        return False


def test_validate_command():
    """Test validate command with valid data."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_file = Path(temp_dir) / "test_data.csv"
            create_test_csv_data(30, filename=str(test_file))

            runner = CliRunner()
            result = runner.invoke(cli, ["validate", "-i", str(test_file)])

            if result.exit_code == 0 and "Data validation passed!" in result.output:
                print("✓ Validate command test passed")
                return True
            else:
                print(f"✗ Validate command test failed: {result.output}")
                return False
    except Exception as e:
        print(f"✗ Validate command test failed with exception: {e}")
        return False


def test_analyze_command_help():
    """Test analyze command help."""
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])

        if result.exit_code == 0 and "--input-csv" in result.output:
            print("✓ Analyze help test passed")
            return True
        else:
            print(f"✗ Analyze help test failed: {result.output}")
            return False
    except Exception as e:
        print(f"✗ Analyze help test failed with exception: {e}")
        return False


def test_analyze_command_missing_args():
    """Test analyze command with missing required arguments."""
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze"])

        if result.exit_code != 0 and "Missing option" in result.output:
            print("✓ Analyze missing args test passed")
            return True
        else:
            print(f"✗ Analyze missing args test failed: {result.output}")
            return False
    except Exception as e:
        print(f"✗ Analyze missing args test failed with exception: {e}")
        return False


def test_analyze_command_invalid_params():
    """Test analyze command with invalid parameters."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_file = Path(temp_dir) / "test_data.csv"
            create_test_csv_data(20, filename=str(test_file))

            runner = CliRunner()

            # Test invalid number of states
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(test_file),
                    "-o",
                    temp_dir,
                    "--n-states",
                    "1",  # Invalid: less than 2
                ],
            )

            if result.exit_code != 0:
                print("✓ Invalid states test passed")
                return True
            else:
                print(f"✗ Invalid states test failed: {result.output}")
                return False
    except Exception as e:
        print(f"✗ Invalid params test failed with exception: {e}")
        return False


def test_analyze_command_basic():
    """Test basic analyze command functionality."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_file = Path(temp_dir) / "test_data.csv"
            create_test_csv_data(
                50, filename=str(test_file)
            )  # Larger dataset for testing

            runner = CliRunner()

            # Test with minimal options
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(test_file),
                    "-o",
                    temp_dir,
                    "--n-states",
                    "2",  # Minimal states
                    "--test-size",
                    "0.2",  # Smaller test set to ensure training data
                    "--random-seed",
                    "42",
                ],
            )

            if result.exit_code == 0:
                print("✓ Basic analyze test passed")
                return True
            else:
                print(f"✗ Basic analyze test failed: {result.output}")
                return False
    except Exception as e:
        print(f"✗ Basic analyze test failed with exception: {e}")
        return False


def test_cli_parameter_validation():
    """Test CLI parameter validation."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_file = Path(temp_dir) / "test_data.csv"
            create_test_csv_data(20, filename=str(test_file))

            runner = CliRunner()

            # Test invalid engine
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(test_file),
                    "-o",
                    temp_dir,
                    "--engine",
                    "invalid_engine",
                ],
            )

            if result.exit_code != 0:
                print("✓ Parameter validation test passed")
                return True
            else:
                print(f"✗ Parameter validation test failed: {result.output}")
                return False
    except Exception as e:
        print(f"✗ Parameter validation test failed with exception: {e}")
        return False


def main():
    """Run simple CLI tests."""
    print("=" * 60)
    print("Simple CLI Integration Test Suite (Task 9)")
    print("=" * 60)

    # Run tests
    tests = [
        ("CLI Import", test_cli_import),
        ("CLI Help", test_cli_help),
        ("CLI Version", test_cli_version),
        ("Validate Command", test_validate_command),
        ("Analyze Help", test_analyze_command_help),
        ("Analyze Missing Args", test_analyze_command_missing_args),
        ("Analyze Invalid Params", test_analyze_command_invalid_params),
        ("Basic Analyze", test_analyze_command_basic),
        ("Parameter Validation", test_cli_parameter_validation),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Simple CLI Integration Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simple CLI tests passed!")
        print("✅ Core CLI functionality is working!")
        return True
    else:
        print("❌ Some CLI tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
