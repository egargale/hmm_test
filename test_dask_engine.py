#!/usr/bin/env python3
"""
Test script for dask_engine functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_dask_engine():
    """Test the dask_engine functionality."""
    print("🧪 Testing Dask Engine...")

    try:
        from processing_engines.dask_engine import (
            compute_with_progress,
            get_dask_cluster_info,
            process_dask,
        )
        from utils import ProcessingConfig, setup_logging

        # Setup logging
        logger = setup_logging(level="INFO")
        logger.info("Starting Dask engine tests")

        # Create test configuration
        config = ProcessingConfig(
            engine_type="dask",
            chunk_size=1000,
            memory_limit_gb=2.0
        )

        # Test Dask processing
        print("\n🔄 Testing Dask processing...")
        ddf = process_dask(
            "BTC.csv",
            config=config,
            scheduler="threads",
            npartitions=4,
            show_progress=True
        )
        print(f"   ✅ Dask DataFrame created with {ddf.npartitions} partitions")
        print(f"   ✅ Columns: {len(ddf.columns)}")

        # Test computation
        print("\n⚙️ Testing computation...")
        df = compute_with_progress(ddf, show_progress=True)
        print("   ✅ Computation completed")
        print(f"   ✅ Result: {len(df)} rows with {len(df.columns)} columns")
        print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

        # Display sample data
        print("\n📋 Sample computed data:")
        sample_cols = ['open', 'high', 'low', 'close', 'volume', 'log_ret', 'obv', 'vwap']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].tail(3))

        # Test cluster monitoring
        print("\n🖥️ Testing cluster monitoring...")
        cluster_info = get_dask_cluster_info()
        if 'error' not in cluster_info:
            print("   ✅ Dask cluster info retrieved")
            print(f"   ✅ Workers: {cluster_info.get('workers', 'N/A')}")
            print(f"   ✅ Total cores: {cluster_info.get('total_cores', 'N/A')}")
            print(f"   ✅ Dashboard: {cluster_info.get('dashboard_link', 'N/A')}")
        else:
            print(f"   ⚠️ Cluster monitoring failed: {cluster_info['error']}")

        # Test optimization
        print("\n⚡ Testing Dask optimization...")
        from processing_engines.dask_engine import optimize_dask_performance
        opt_settings = optimize_dask_performance(scheduler="threads", memory_limit="1GB")
        print(f"   ✅ Dask optimization applied: {opt_settings}")

        print("\n✅ Dask engine test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Dask engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dask_engine()
    sys.exit(0 if success else 1)
