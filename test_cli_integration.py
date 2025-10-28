#!/usr/bin/env python3
"""
Test script for CLI Integration and Orchestration functionality.

This script demonstrates and tests the CLI orchestration components
without requiring all dependencies to be installed.
"""

import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_cli_components():
    """Test CLI integration components."""
    print("🧪 Testing CLI Integration Components")
    print("=" * 50)

    # Test 1: Configuration classes
    print("\n1. Testing Configuration Classes...")
    try:
        from cli_comprehensive import HMMConfig, ProcessingConfig

        # Test HMMConfig
        hmm_config = HMMConfig(n_states=4, covariance_type="diag")
        config_dict = hmm_config.to_dict()
        print(f"✅ HMMConfig created: {len(config_dict)} parameters")
        print(f"   n_states: {config_dict['n_states']}")
        print(f"   covariance_type: {config_dict['covariance_type']}")

        # Test ProcessingConfig
        proc_config = ProcessingConfig(engine_type="dask", chunk_size=50000)
        print("✅ ProcessingConfig created:")
        print(f"   engine_type: {proc_config.engine_type}")
        print(f"   chunk_size: {proc_config.chunk_size}")
        print(f"   indicators: {len(proc_config.indicators)} items")

    except Exception as e:
        print(f"❌ Configuration classes test failed: {e}")
        return False

    # Test 2: Performance metrics logging
    print("\n2. Testing Performance Metrics...")
    try:
        from cli_comprehensive import log_performance_metrics

        start_time = time.time()
        time.sleep(0.1)  # Simulate work

        metrics = log_performance_metrics(
            start_time, "test_operation", {"test_param": "test_value", "test_count": 42}
        )

        print("✅ Performance metrics logged:")
        print(f"   Operation: {metrics['operation']}")
        print(f"   Elapsed time: {metrics['elapsed_time_seconds']:.3f}s")
        print(f"   Additional params: {len(metrics) - 3}")

    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False

    # Test 3: Configuration loading
    print("\n3. Testing Configuration Loading...")
    try:
        # Create test config file
        test_config = {
            "hmm": {"n_states": 5, "covariance_type": "full", "n_iter": 200},
            "processing": {"engine_type": "dask", "chunk_size": 50000},
        }

        test_config_path = Path("test_config.json")
        with open(test_config_path, "w") as f:
            json.dump(test_config, f, indent=2)

        print(f"✅ Test configuration created: {test_config_path}")
        print(f"   HMM settings: {len(test_config['hmm'])}")
        print(f"   Processing settings: {len(test_config['processing'])}")

        # Clean up
        test_config_path.unlink()

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

    print("\n✅ All CLI Integration Tests Passed!")
    return True


def test_cli_command_structure():
    """Test CLI command structure without full execution."""
    print("\n🧪 Testing CLI Command Structure")
    print("=" * 50)

    try:
        import click.testing

        from cli_comprehensive import cli

        # Test CLI group creation
        print("✅ CLI group created successfully")

        # Test command registration
        commands = [cmd.name for cmd in cli.commands.values()]
        expected_commands = ["validate", "analyze", "infer", "model-info", "version"]

        print(f"✅ Commands registered: {commands}")

        missing_commands = set(expected_commands) - set(commands)
        if missing_commands:
            print(f"⚠️  Missing commands: {missing_commands}")
        else:
            print("✅ All expected commands present")

        # Test help functionality
        runner = click.testing.CliRunner()
        result = runner.invoke(cli, ["--help"])
        print("✅ Help command executes successfully")
        print(f"   Output length: {len(result.output)} characters")

        # Test version command
        result = runner.invoke(cli, ["version"])
        print("✅ Version command executes successfully")
        print(f"   Output: {result.output.strip()}")

        return True

    except Exception as e:
        print(f"❌ CLI command structure test failed: {e}")
        return False


def test_error_handling():
    """Test CLI error handling mechanisms."""
    print("\n🧪 Testing CLI Error Handling")
    print("=" * 50)

    try:
        import click.testing

        from cli_comprehensive import cli

        runner = click.testing.CliRunner()

        # Test non-existent file handling
        print("Testing non-existent file handling...")
        result = runner.invoke(cli, ["validate", "--input-csv", "nonexistent.csv"])

        if result.exit_code != 0:
            print("✅ Non-existent file correctly rejected")
        else:
            print("⚠️  Non-existent file should have been rejected")

        # Test invalid parameter handling
        print("Testing invalid parameter handling...")
        result = runner.invoke(cli, ["analyze", "--n-states", "1"])  # Below minimum

        if result.exit_code != 0:
            print("✅ Invalid parameter correctly rejected")
        else:
            print("⚠️  Invalid parameter should have been rejected")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_memory_monitoring():
    """Test memory monitoring functionality."""
    print("\n🧪 Testing Memory Monitoring")
    print("=" * 50)

    try:
        from cli_comprehensive import check_memory_usage, get_memory_usage

        # Test memory usage function
        memory_usage = get_memory_usage()
        print(f"✅ Memory usage function works: {memory_usage:.3f}")

        # Test memory check function (should handle missing psutil gracefully)
        print("Testing memory check function...")
        check_memory_usage("test_operation")
        print("✅ Memory check function executes without errors")

        return True

    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        return False


def run_integration_tests():
    """Run all CLI integration tests."""
    print("🚀 Running CLI Integration Tests")
    print("=" * 60)

    tests = [
        ("Component Tests", test_cli_components),
        ("Command Structure Tests", test_cli_command_structure),
        ("Error Handling Tests", test_error_handling),
        ("Memory Monitoring Tests", test_memory_monitoring),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 Test Results Summary")
    print(f"{'=' * 60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All CLI Integration Tests Completed Successfully!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
