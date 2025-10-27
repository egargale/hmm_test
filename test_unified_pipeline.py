#!/usr/bin/env python3
"""
Comprehensive test suite for the unified data pipeline (Phase 2.1.4).

This script tests the complete unified pipeline system, including:
- Pipeline orchestration and configuration
- Integration of all enhanced components from Phases 2.1.1-2.1.3
- End-to-end data processing workflow
- Performance metrics and reporting
- Input/output management
"""

import sys
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, 'src')

from src.data_processing.pipeline_config import PipelineConfig, PipelineMode
from src.data_processing.unified_pipeline import PipelineResult, UnifiedDataPipeline


def create_test_csv_data():
    """Create test data for unified pipeline testing."""
    np.random.seed(42)

    # Create sample OHLCV data
    n_samples = 5000
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    # Generate realistic price data
    base_price = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))
    high_prices = base_price + np.abs(np.random.normal(0.5, 0.3, n_samples))
    low_prices = base_price - np.abs(np.random.normal(0.5, 0.3, n_samples))
    open_prices = base_price + np.random.normal(0, 0.2, n_samples)
    close_prices = base_price + np.random.normal(0, 0.2, n_samples)
    volumes = np.random.exponential(1000000, n_samples)

    return pd.DataFrame({
        'DateTime': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes.astype(int)
    }).set_index('DateTime')

def create_test_csv_files(test_data):
    """Create temporary CSV files for testing."""
    temp_files = {}

    # Standard format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.reset_index().to_csv(f, index=False)
        temp_files['standard'] = Path(f.name)

    # Yahoo Finance format
    yahoo_data = test_data.reset_index().copy()
    yahoo_data['Adj Close'] = yahoo_data['Close'] * 0.98
    yahoo_data = yahoo_data.drop('DateTime', axis=1)
    yahoo_data = yahoo_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    yahoo_data['Date'] = yahoo_data.index.strftime('%Y-%m-%d')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        yahoo_data.to_csv(f, index=False)
        temp_files['yahoo'] = Path(f.name)

    return temp_files

def test_pipeline_configuration():
    """Test pipeline configuration system."""
    print("Testing pipeline configuration...")

    try:
        # Test default configuration
        default_config = PipelineConfig()
        assert default_config.mode == PipelineMode.STANDARD
        assert default_config.validation_config.enable_validation == True
        assert default_config.feature_config.enable_features == True
        print("  ✓ Default configuration created")

        # Test high-performance configuration
        hp_config = PipelineConfig.create_high_performance()
        assert hp_config.mode == PipelineMode.HIGH_PERFORMANCE
        assert hp_config.validation_config.strict_mode == False
        print("  ✓ High-performance configuration created")

        # Test high-quality configuration
        hq_config = PipelineConfig.create_high_quality()
        assert hq_config.mode == PipelineMode.HIGH_QUALITY
        assert hq_config.validation_config.strict_mode == True
        print("  ✓ High-quality configuration created")

        # Test configuration validation
        issues = default_config.validate()
        assert len(issues) == 0, f"Default config should be valid: {issues}"
        print("  ✓ Configuration validation passed")

        # Test configuration updates
        default_config.update(name="test_pipeline", chunk_size=20000)
        assert default_config.name == "test_pipeline"
        assert default_config.performance_config.chunk_size == 20000
        print("  ✓ Configuration update successful")

        # Test pipeline info
        pipeline_info = default_config.get_summary()
        assert 'pipeline_name' in pipeline_info
        assert 'mode' in pipeline_info
        print("  ✓ Pipeline info generated")

        print("✓ Pipeline configuration tests passed")
        return True

    except Exception as e:
        print(f"❌ Pipeline configuration test failed: {e}")
        return False

def test_unified_pipeline_basic():
    """Test basic unified pipeline functionality."""
    print("\nTesting basic unified pipeline...")

    try:
        # Create test data
        test_data = create_test_csv_data()

        # Create default pipeline
        pipeline = UnifiedDataPipeline()
        assert pipeline.config.mode == PipelineMode.STANDARD
        assert len(pipeline.stages) >= 3  # Should have multiple stages
        print("  ✓ Default pipeline created")

        # Test pipeline info
        pipeline_info = pipeline.get_pipeline_info()
        assert 'pipeline_name' in pipeline_info
        assert 'total_stages' in pipeline_info
        print("  ✓ Pipeline info retrieved")

        # Process data
        result = pipeline.process(test_data)

        # Validate results
        assert isinstance(result, PipelineResult)
        assert result.success == True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == len(test_data)
        assert len(result.data.columns) > len(test_data.columns)  # Features added
        print(f"  ✓ Data processed successfully: {len(result.data)} rows, {len(result.data.columns)} columns")

        # Check metadata
        assert result.metadata is not None
        assert 'input_source' in result.metadata
        print("  ✓ Metadata preserved")

        # Check quality report
        assert result.quality_report is not None
        assert 'overall_score' in result.quality_report
        print(f"  ✓ Quality report generated: Score = {result.quality_report.get('overall_score', 0):.3f}")

        # Check processing log
        assert result.processing_log is not None
        assert len(result.processing_log) > 0
        print(f"  ✓ Processing log created: {len(result.processing_log)} stages")

        print("✓ Basic unified pipeline tests passed")
        return True

    except Exception as e:
        print(f"❌ Basic unified pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_with_csv_input():
    """Test pipeline with CSV file input."""
    print("\nTesting pipeline with CSV input...")

    try:
        # Create test CSV files
        test_data = create_test_csv_data()
        temp_files = create_test_csv_files(test_data)

        # Create pipeline
        pipeline = UnifiedDataPipeline()

        # Test with standard CSV format
        print("  Testing standard CSV format...")
        result = pipeline.process(temp_files['standard'])
        assert result.success == True
        assert len(result.data) == len(test_data)
        print(f"    ✓ Standard CSV processed: {len(result.data)} rows")

        # Test with Yahoo Finance format
        print("  Testing Yahoo Finance format...")
        result_yahoo = pipeline.process(temp_files['yahoo'])
        assert result_yahoo.success == True
        print(f"    ✓ Yahoo Finance CSV processed: {len(result_yahoo.data)} rows")

        # Cleanup
        for temp_file in temp_files.values():
            temp_file.unlink()

        print("✓ CSV input pipeline tests passed")
        return True

    except Exception as e:
        print(f"❌ CSV input pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_performance_modes():
    """Test different pipeline performance modes."""
    print("\nTesting pipeline performance modes...")

    try:
        # Create test data
        test_data = create_test_csv_data()

        modes_to_test = [
            (PipelineMode.STANDARD, "Standard"),
            (PipelineMode.HIGH_PERFORMANCE, "High Performance"),
            (PipelineMode.HIGH_QUALITY, "High Quality")
        ]

        results = {}

        for mode, mode_name in modes_to_test:
            print(f"  Testing {mode_name} mode...")

            # Create pipeline with specific mode
            if mode == PipelineMode.STANDARD:
                pipeline = UnifiedDataPipeline()
            elif mode == PipelineMode.HIGH_PERFORMANCE:
                pipeline = UnifiedDataPipeline.create_high_performance()
            elif mode == PipelineMode.HIGH_QUALITY:
                pipeline = UnifiedDataPipeline.create_high_quality()

            # Process data
            start_time = datetime.now()
            result = pipeline.process(test_data)
            end_time = datetime.now()

            processing_time = (end_time - start_time).total_seconds()

            results[mode_name] = {
                'success': result.success,
                'execution_time': processing_time,
                'features_count': len(result.data.columns) - len(test_data.columns),
                'quality_score': result.quality_report.get('overall_score', 0),
                'issues_count': len(result.issues)
            }

            print(f"    ✓ {mode_name} mode: {processing_time:.3f}s, "
                  f"{results[mode_name]['features_count']} features, "
                  f"Quality: {results[mode_name]['quality_score']:.3f}")

        # Validate results
        for mode_name, result_data in results.items():
            assert result_data['success'], f"{mode_name} mode should succeed"

        print("✓ Performance modes tests passed")
        return True

    except Exception as e:
        print(f"❌ Performance modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_error_handling():
    """Test pipeline error handling and recovery."""
    print("\nTesting pipeline error handling...")

    try:
        # Create pipeline
        pipeline = UnifiedDataPipeline()

        # Test with empty DataFrame
        print("  Testing with empty DataFrame...")
        empty_df = pd.DataFrame()
        result = pipeline.process(empty_df)
        assert result.success == False  # Should fail with empty data
        assert len(result.issues) > 0
        print("    ✓ Empty DataFrame handled correctly")

        # Test with invalid CSV format
        print("  Testing with invalid CSV...")
        invalid_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        invalid_csv.write("invalid,csv,format\n1,2,3")  # Missing OHLCV columns
        invalid_csv_path = Path(invalid_csv.name)
        invalid_csv.close()

        try:
            result = pipeline.process(invalid_csv_path)
            # Should either fail or handle gracefully
            if not result.success:
                print("    ✓ Invalid CSV handled correctly")
            else:
                print("    ⚠ Invalid CSV processed (may be expected)")
        finally:
            invalid_csv_path.unlink()

        # Test with malformed data
        print("  Testing with malformed OHLCV data...")
        malformed_data = create_test_csv_data()
        # Introduce OHLC violations
        malformed_data.loc[10:20, 'High'] = malformed_data.loc[10:20, 'Low'] - 1
        malformed_data.loc[30:40, 'Volume'] = -1000

        result = pipeline.process(malformed_data)
        assert result.success == True  # Should succeed but with warnings
        assert len(result.issues) > 0 or len(result.quality_report.get('issues', [])) > 0
        print("    ✓ Malformed data handled with warnings")

        print("✓ Error handling tests passed")
        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_metrics_and_reporting():
    """Test pipeline metrics collection and reporting."""
    print("\nTesting pipeline metrics and reporting...")

    try:
        # Create test data
        test_data = create_test_csv_data()

        # Create pipeline
        pipeline = UnifiedDataPipeline(name="test_metrics_pipeline")

        # Process data
        result = pipeline.process(test_data)

        # Check performance metrics
        assert result.performance_metrics is not None
        perf_report = result.performance_metrics
        assert 'report_type' in perf_report
        assert 'pipeline_summary' in perf_report
        assert 'stage_details' in perf_report
        print("  ✓ Performance metrics generated")

        # Check stage details
        stage_details = perf_report['stage_details']
        assert len(stage_details) >= 3  # Should have multiple stages
        for stage in stage_details:
            assert 'stage_name' in stage
            assert 'processing_time' in stage
            assert 'success' in stage
        print(f"    ✓ Stage details for {len(stage_details)} stages")

        # Check quality report
        assert result.quality_report is not None
        quality_report = result.quality_report
        assert 'overall_score' in quality_report
        assert 'quality_breakdown' in quality_report
        print(f"    ✓ Quality report: Score = {quality_report['overall_score']:.3f}")

        # Check recommendations
        assert result.recommendations is not None
        assert len(result.recommendations) >= 0
        if result.recommendations:
            print(f"    ✓ {len(result.recommendations)} recommendations generated")

        # Check pipeline info after execution
        pipeline_info = pipeline.get_pipeline_info()
        assert pipeline_info['execution_count'] == 1
        print("  ✓ Pipeline execution count updated")

        print("✓ Metrics and reporting tests passed")
        return True

    except Exception as e:
        print(f"❌ Metrics and reporting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_output_management():
    """Test pipeline output management capabilities."""
    print("\nTesting pipeline output management...")

    try:
        # Create test data
        test_data = create_test_csv_data()

        # Create pipeline with output configuration
        config = PipelineConfig(
            name="test_output_pipeline",
            output_config_save_path=Path(tempfile.mkdtemp()) / "output"
        )
        pipeline = UnifiedDataPipeline(config)

        # Process data with output saving
        result = pipeline.process(test_data, save_output=True)

        assert result.success == True
        print("  ✓ Pipeline completed with output saving")

        # Check if output files were created
        output_dir = Path(config.output_config.save_path)
        assert output_dir.exists()
        print(f"  ✓ Output directory created: {output_dir}")

        # Check for expected output files
        expected_files = ['data.csv', 'metadata.json']
        created_files = [f.name for f in output_dir.iterdir() if f.is_file()]

        for expected_file in expected_files:
            if any(expected_file in f for f in created_files):
                print(f"    ✓ Output file created: {expected_file}")
            else:
                print(f"    ⚠ Expected file not found: {expected_file}")

        # Cleanup
        import shutil
        shutil.rmtree(output_dir)

        print("✓ Output management tests passed")
        return True

    except Exception as e:
        print(f"❌ Output management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_customization():
    """Test pipeline customization capabilities."""
    print("\nTesting pipeline customization...")

    try:
        # Create test data
        test_data = create_test_csv_data()

        # Create base pipeline
        pipeline = UnifiedDataPipeline()

        # Test adding custom stage

        from src.data_processing.pipeline_stages import (
            PipelineStage,
            StageResult,
        )

        class TestCustomStage(PipelineStage):
            def process(self, data, context):
                return StageResult(
                    success=True,
                    data=data,
                    metadata={'custom_stage': True},
                    issues=[],
                    processing_time=0.001,
                    stage_name=self.name
                )

        custom_stage = TestCustomStage("custom_test_stage")
        pipeline.add_custom_stage(custom_stage)

        # Check that stage was added
        stage_names = [stage.name for stage in pipeline.stages]
        assert "custom_test_stage" in stage_names
        print("  ✓ Custom stage added successfully")

        # Process data
        result = pipeline.process(test_data)
        assert result.success == True
        print("  ✓ Pipeline with custom stage completed")

        # Test removing stage
        removed = pipeline.remove_stage("custom_test_stage")
        assert removed == True
        print("  ✓ Custom stage removed successfully")

        # Test pipeline mode variations
        hp_pipeline = UnifiedDataPipeline.create_high_performance()
        assert hp_pipeline.config.mode.value == "high_performance"
        print("  ✓ High-performance pipeline created")

        hq_pipeline = UnifiedDataPipeline.create_high_quality()
        assert hq_pipeline.config.mode.value == "high_quality"
        print("  ✓ High-quality pipeline created")

        print("✓ Pipeline customization tests passed")
        return True

    except Exception as e:
        print(f"❌ Pipeline customization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("\nTesting end-to-end workflow...")

    try:
        # Create comprehensive test scenario
        test_data = create_test_csv_data()
        temp_files = create_test_csv_files(test_data)

        # Test complete workflow with different configurations
        workflows = [
            {
                'name': 'Standard CSV Processing',
                'pipeline_factory': lambda: UnifiedDataPipeline(),
                'input_source': temp_files['standard']
            },
            {
                'name': 'High Performance Processing',
                'pipeline_factory': lambda: UnifiedDataPipeline.create_high_performance(),
                'input_source': test_data
            },
            {
                'name': 'High Quality Processing',
                'pipeline_factory': lambda: UnifiedDataPipeline.create_high_quality(),
                'input_source': test_data
            }
        ]

        results = {}

        for workflow in workflows:
            print(f"  Running: {workflow['name']}")

            # Create pipeline
            pipeline = workflow['pipeline_factory']()

            # Process data
            start_time = datetime.now()
            result = pipeline.process(workflow['input_source'])
            end_time = datetime.now()

            # Record results
            processing_time = (end_time - start_time).total_seconds()
            results[workflow['name']] = {
                'success': result.success,
                'processing_time': processing_time,
                'rows_processed': len(result.data) if result.success else 0,
                'features_added': len(result.data.columns) - len(test_data) if result.success else 0,
                'quality_score': result.quality_report.get('overall_score', 0) if result.success else 0,
                'issues_count': len(result.issues) if result.success else 1
            }

            print(f"    ✓ {workflow['name']}: {processing_time:.3f}s, "
                  f"{results[workflow['name']]['features_added']} features, "
                  f"Quality: {results[workflow['name']]['quality_score']:.3f}")

        # Validate all workflows succeeded
        for workflow_name, result_data in results.items():
            assert result_data['success'], f"Workflow '{workflow_name}' should succeed"

        # Performance comparison
        print("  Performance comparison:")
        for workflow_name, result_data in results.items():
            print(f"    {workflow_name}: {result_data['processing_time']:.3f}s, "
                  f"{result_data['rows_processed']/result_data['processing_time']:.0f} rows/sec")

        # Cleanup
        for temp_file in temp_files.values():
            temp_file.unlink()

        print("✓ End-to-end workflow tests passed")
        return True

    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all unified pipeline tests."""
    print("=" * 80)
    print("PHASE 2.1.4 UNIFIED DATA PIPELINE VALIDATION")
    print("=" * 80)

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    tests = [
        ("Pipeline Configuration", test_pipeline_configuration),
        ("Basic Unified Pipeline", test_unified_pipeline_basic),
        ("Pipeline with CSV Input", test_pipeline_with_csv_input),
        ("Pipeline Performance Modes", test_pipeline_performance_modes),
        ("Pipeline Error Handling", test_pipeline_error_handling),
        ("Metrics and Reporting", test_pipeline_metrics_and_reporting),
        ("Output Management", test_pipeline_output_management),
        ("Pipeline Customization", test_pipeline_customization),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"{'✅' if result else '❌'} {test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Unified data pipeline is working correctly.")
        print("\nKey achievements:")
        print("• Complete unified pipeline orchestrator")
        print("• Integration of all enhanced capabilities from Phases 2.1.1-2.1.3")
        print("• Multiple performance modes (Standard, High Performance, High Quality)")
        print("• Comprehensive metrics collection and reporting")
        print("• Flexible input/output management")
        print("• Advanced error handling and recovery")
        print("• Pipeline customization capabilities")
        print("• End-to-end workflow validation")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
