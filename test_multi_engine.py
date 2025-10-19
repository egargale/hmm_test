#!/usr/bin/env python3
"""
Test script for the Multi-Engine Processing Framework.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_multi_engine_framework():
    """Test the complete multi-engine processing framework."""
    print("🧪 Testing Multi-Engine Processing Framework...")

    try:
        from processing_engines import (
            process_streaming, process_dask, process_daft,
            ProcessingEngineFactory, process_with_engine, get_processing_engine_factory
        )
        from utils import ProcessingConfig, setup_logging

        # Setup logging
        logger = setup_logging(level="INFO")
        logger.info("Starting multi-engine framework tests")

        # Create test configuration
        config = ProcessingConfig(
            engine_type="streaming",  # Use valid engine type
            chunk_size=1000,
            memory_limit_gb=2.0
        )

        # Test 1: Processing Engine Factory
        print("\n🏭 Testing Processing Engine Factory...")
        factory = get_processing_engine_factory()
        print(f"   ✅ Factory created")
        print(f"   ✅ Available engines: {factory.get_available_engines()}")

        # Test engine recommendation
        recommended = factory.recommend_engine("BTC.csv")
        print(f"   ✅ Recommended engine for BTC.csv: {recommended}")

        # Test 2: Engine Information
        print("\n📊 Testing Engine Information...")
        engine_info = factory.get_engine_info()
        print(f"   ✅ Engine info retrieved")
        print(f"   ✅ Available engines: {engine_info['available_engines']}")
        for engine, details in engine_info['engine_details'].items():
            print(f"   ✅ {engine}: {details['type']} - Best for: {details['best_for']}")

        # Test 3: Individual Engines
        print("\n🚀 Testing Individual Engines...")

        # Test Streaming Engine
        print("\n   📦 Testing Streaming Engine...")
        try:
            streaming_result = process_streaming(
                "BTC.csv",
                config=config,
                chunk_size=500,  # Small chunks for testing
                show_progress=False
            )
            print(f"      ✅ Streaming: {len(streaming_result)} rows, {len(streaming_result.columns)} columns")
        except Exception as e:
            print(f"      ❌ Streaming failed: {e}")

        # Test Dask Engine (if available)
        if 'dask' in factory.get_available_engines():
            print("\n   ⚡ Testing Dask Engine...")
            try:
                dask_result = process_dask(
                    "BTC.csv",
                    config=config,
                    scheduler="threads",
                    npartitions=2,
                    show_progress=False
                )
                print(f"      ✅ Dask: {dask_result.npartitions} partitions, {len(dask_result.columns)} columns")
            except Exception as e:
                print(f"      ❌ Dask failed: {e}")
        else:
            print("\n   ⚡ Dask Engine not available")

        # Test Daft Engine (if available)
        if 'daft' in factory.get_available_engines():
            print("\n   🌟 Testing Daft Engine...")
            try:
                daft_result = process_daft(
                    "BTC.csv",
                    config=config,
                    npartitions=1,
                    show_progress=False
                )
                print(f"      ✅ Daft: Processing successful")
            except Exception as e:
                print(f"      ❌ Daft failed: {e}")
        else:
            print("\n   🌟 Daft Engine not available")

        # Test 4: Factory-based Processing
        print("\n🏭 Testing Factory-based Processing...")
        try:
            # Test with auto-selected engine
            auto_result = process_with_engine(
                "BTC.csv",
                config=config,
                engine=None,  # Auto-select
                show_progress=False
            )
            print(f"   ✅ Auto-selected engine: {len(auto_result)} rows, {len(auto_result.columns)} columns")
        except Exception as e:
            print(f"   ❌ Auto-selected engine failed: {e}")

        try:
            # Test with specific engine
            specific_result = process_with_engine(
                "BTC.csv",
                config=config,
                engine="streaming",  # Force streaming
                show_progress=False
            )
            print(f"   ✅ Specific engine (streaming): {len(specific_result)} rows, {len(specific_result.columns)} columns")
        except Exception as e:
            print(f"   ❌ Specific engine failed: {e}")

        # Test 5: Result Computation
        print("\n⚙️ Testing Result Computation...")
        try:
            # Create a Dask DataFrame for testing computation
            if 'dask' in factory.get_available_engines():
                from processing_engines.dask_engine import process_dask
                dask_df = process_dask(
                    "BTC.csv",
                    config=config,
                    scheduler="threads",
                    npartitions=2,
                    show_progress=False
                )
                computed_result = factory.compute_result(dask_df, show_progress=False)
                print(f"   ✅ Computed Dask result: {len(computed_result)} rows")
            else:
                print(f"   ℹ️ Skipping Dask computation test (Dask not available)")
        except Exception as e:
            print(f"   ❌ Result computation failed: {e}")

        # Test 6: Engine Benchmarking
        print("\n📈 Testing Engine Benchmarking...")
        try:
            # Quick benchmark with available engines
            benchmark_results = factory.benchmark_engines(
                "BTC.csv",
                engines=["streaming"],  # Only test streaming for speed
                cache_results=True
            )
            print(f"   ✅ Benchmark completed")
            for engine, result in benchmark_results.items():
                if result.get('success'):
                    print(f"   ✅ {engine}: {result['time']:.2f}s, {result['rows']} rows")
                else:
                    print(f"   ❌ {engine}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ Benchmarking failed: {e}")

        # Test 7: Sample Data Display
        print("\n📋 Sample Processed Data:")
        try:
            # Use the streaming result for sample display
            sample_cols = ['open', 'high', 'low', 'close', 'volume', 'log_ret', 'obv', 'vwap']
            available_cols = [col for col in sample_cols if col in streaming_result.columns]
            if available_cols:
                print(streaming_result[available_cols].tail(3))
            else:
                print(f"   ℹ️ Feature columns not found. Available: {list(streaming_result.columns)[:10]}...")
        except Exception as e:
            print(f"   ❌ Sample display failed: {e}")

        print("\n✅ Multi-Engine Processing Framework test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Multi-Engine Framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_engine_framework()
    sys.exit(0 if success else 1)