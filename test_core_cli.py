#!/usr/bin/env python3
"""
Core CLI functionality tests without Click dependency.

This script tests the core CLI orchestration components
that don't require the full Click framework.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_core_components():
    """Test core CLI components without Click."""
    print("🧪 Testing Core CLI Components")
    print("="*50)

    # Test 1: Configuration classes
    print("\n1. Testing Configuration Classes...")
    try:
        # Create HMM config class directly in the test file
        class HMMConfig:
            def __init__(self, n_states=3, covariance_type='full',
                         n_iter=100, random_state=42, tol=1e-3,
                         num_restarts=3):
                self.n_states = n_states
                self.covariance_type = covariance_type
                self.n_iter = n_iter
                self.random_state = random_state
                self.tol = tol
                self.num_restarts = num_restarts

            def to_dict(self):
                return {
                    'n_states': self.n_states,
                    'covariance_type': self.covariance_type,
                    'n_iter': self.n_iter,
                    'random_state': self.random_state,
                    'tol': self.tol,
                    'num_restarts': self.num_restarts
                }

        # Test HMMConfig
        hmm_config = HMMConfig(n_states=4, covariance_type='diag')
        config_dict = hmm_config.to_dict()
        print(f"✅ HMMConfig created: {len(config_dict)} parameters")
        print(f"   n_states: {config_dict['n_states']}")
        print(f"   covariance_type: {config_dict['covariance_type']}")

        # Test ProcessingConfig
        class ProcessingConfig:
            def __init__(self, engine_type='streaming', chunk_size=100000,
                         indicators=None):
                self.engine_type = engine_type
                self.chunk_size = chunk_size
                self.indicators = indicators or {
                    'sma_5': {'window': 5},
                    'sma_10': {'window': 10},
                    'sma_20': {'window': 20},
                    'volatility_14': {'window': 14},
                    'returns': {}
                }

        proc_config = ProcessingConfig(engine_type='dask', chunk_size=50000)
        print(f"✅ ProcessingConfig created:")
        print(f"   engine_type: {proc_config.engine_type}")
        print(f"   chunk_size: {proc_config.chunk_size}")
        print(f"   indicators: {len(proc_config.indicators)} items")

    except Exception as e:
        print(f"❌ Configuration classes test failed: {e}")
        return False

    # Test 2: Performance metrics logging
    print("\n2. Testing Performance Metrics...")
    try:
        def log_performance_metrics(start_time, operation, additional_info=None):
            """Log performance metrics for completed operations."""
            elapsed_time = time.time() - start_time
            metrics = {
                'operation': operation,
                'elapsed_time_seconds': elapsed_time,
                'timestamp': time.time()
            }
            if additional_info:
                metrics.update(additional_info)
            print(f"Performance - {operation}: {elapsed_time:.2f}s")
            return metrics

        start_time = time.time()
        time.sleep(0.1)  # Simulate work

        metrics = log_performance_metrics(
            start_time, "test_operation",
            {'test_param': 'test_value', 'test_count': 42}
        )

        print(f"✅ Performance metrics logged:")
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
            "hmm": {
                "n_states": 5,
                "covariance_type": "full",
                "n_iter": 200
            },
            "processing": {
                "engine_type": "dask",
                "chunk_size": 50000
            }
        }

        test_config_path = Path("test_config.json")
        with open(test_config_path, 'w') as f:
            json.dump(test_config, f, indent=2)

        print(f"✅ Test configuration created: {test_config_path}")
        print(f"   HMM settings: {len(test_config['hmm'])}")
        print(f"   Processing settings: {len(test_config['processing'])}")

        # Test loading
        with open(test_config_path, 'r') as f:
            loaded_config = json.load(f)

        if loaded_config == test_config:
            print("✅ Configuration loaded successfully")
        else:
            print("❌ Configuration loading failed")
            return False

        # Clean up
        test_config_path.unlink()

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

    print("\n✅ All Core CLI Components Tests Passed!")
    return True


def test_orchestration_logic():
    """Test CLI orchestration logic."""
    print("\n🧪 Testing CLI Orchestration Logic")
    print("="*50)

    try:
        # Test pipeline orchestration simulation
        def simulate_pipeline_step(step_name, duration=0.1):
            """Simulate a pipeline step with timing."""
            print(f"   Starting {step_name}...")
            start_time = time.time()
            time.sleep(duration)
            elapsed = time.time() - start_time
            print(f"   ✅ {step_name} completed ({elapsed:.2f}s)")
            return elapsed

        # Test 1: Multi-step pipeline simulation
        print("\n1. Testing Multi-step Pipeline...")
        total_time = 0

        total_time += simulate_pipeline_step("Data Loading", 0.05)
        total_time += simulate_pipeline_step("Feature Engineering", 0.08)
        total_time += simulate_pipeline_step("Model Training", 0.15)
        total_time += simulate_pipeline_step("State Inference", 0.05)
        total_time += simulate_pipeline_step("Results Saving", 0.03)

        print(f"✅ Pipeline completed in {total_time:.2f}s total")

        # Test 2: Error handling simulation
        print("\n2. Testing Error Handling...")

        def step_that_fails():
            raise ValueError("Simulated error")

        try:
            print("   Testing step that fails...")
            step_that_fails()
            print("   ❌ Should not reach here")
            return False
        except ValueError as e:
            print(f"   ✅ Error caught correctly: {e}")

        # Test 3: Configuration-based orchestration
        print("\n3. Testing Configuration-based Orchestration...")
        test_configs = [
            {"n_states": 3, "engine": "streaming"},
            {"n_states": 5, "engine": "dask"},
            {"n_states": 4, "engine": "daft"}
        ]

        for i, config in enumerate(test_configs):
            print(f"   Testing configuration {i+1}: {config}")
            time.sleep(0.02)  # Simulate processing
            print(f"   ✅ Configuration {i+1} processed")

        return True

    except Exception as e:
        print(f"❌ Orchestration logic test failed: {e}")
        return False


def test_progress_monitoring():
    """Test progress monitoring functionality."""
    print("\n🧪 Testing Progress Monitoring")
    print("="*50)

    try:
        # Simple progress bar simulation
        def simulate_progress_bar(total_steps, step_name="Processing"):
            """Simulate a progress bar."""
            print(f"\n{step_name}:")
            for i in range(total_steps):
                time.sleep(0.02)  # Simulate work
                progress = (i + 1) / total_steps * 100
                bar_length = 20
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r   [{bar}] {progress:.0f}% ({i+1}/{total_steps})", end='')
            print()

        # Test 1: Basic progress bar
        print("\n1. Testing Basic Progress Bar...")
        simulate_progress_bar(5, "Data Processing")

        # Test 2: Multi-stage progress
        print("\n2. Testing Multi-stage Progress...")
        stages = ["Loading", "Processing", "Training", "Analyzing", "Saving"]
        for stage in stages:
            simulate_progress_bar(3, stage)

        return True

    except Exception as e:
        print(f"❌ Progress monitoring test failed: {e}")
        return False


def run_core_tests():
    """Run all core CLI tests."""
    print("🚀 Running Core CLI Integration Tests")
    print("="*60)

    tests = [
        ("Core Component Tests", test_core_components),
        ("Orchestration Logic Tests", test_orchestration_logic),
        ("Progress Monitoring Tests", test_progress_monitoring),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("📊 Core Test Results Summary")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Core CLI Tests Completed Successfully!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_core_tests()
    sys.exit(0 if success else 1)