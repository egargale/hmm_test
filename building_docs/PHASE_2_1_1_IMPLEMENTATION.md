# Phase 2.1.1: Migrate main.py Core Functionity
## Implementation Plan and Progress

**Objective**: Migrate core functionality from `main.py` to enhanced `src` directory structure while maintaining backward compatibility and adding enhanced features.

**Target Files**:
- Source: `main.py` (410 lines)
- Target: Enhanced `src/` modules with new unified interfaces

---

## 1. Migration Mapping Analysis

### 1.1 Core Functions to Migrate

| Function | Lines | Target Module | Enhancement Strategy |
|----------|-------|--------------|---------------------|
| `add_features()` | 33-86 | `src/data_processing/feature_engineering.py` | Enhance with configurable indicators |
| `stream_features()` | 272-352 | `src/data_processing/streaming_processor.py` | NEW: Enhanced streaming with memory optimization |
| `simple_backtest()` | 250-260 | `src/backtesting/strategy_engine.py` | Enhance with multiple strategies |
| `perf_metrics()` | 262-267 | `src/backtesting/performance_metrics.py` | Enhance with additional metrics |
| `main()` | 95-244 | `src/pipelines/hmm_pipeline.py` | NEW: Unified pipeline interface |

### 1.2 Configuration and CLI

| Component | Lines | Target Module | Enhancement Strategy |
|-----------|-------|--------------|---------------------|
| Argument parsing | 357-407 | `src/cli/hmm_commands.py` | NEW: Enhanced CLI with subcommands |
| Logging setup | 25-28 | `src/utils/logging_config.py` | Already exists - enhance |
| Model persistence | 130-176 | `src/model_training/model_persistence.py` | Already exists - enhance |

---

## 2. Enhanced Architecture Design

### 2.1 New Unified Pipeline Interface

```python
# src/pipelines/hmm_pipeline.py
class HMMPipeline:
    """Unified HMM training and inference pipeline"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config.features)
        self.trainer = HMMTrainer(config.training)
        self.persistence = ModelPersistence(config.persistence)

    async def run(self, data_path: Path) -> PipelineResult:
        """Complete pipeline execution"""
        # 1. Load and process data
        # 2. Feature engineering
        # 3. Model training/inference
        # 4. Results persistence
        # 5. Optional backtesting
        pass
```

### 2.2 Enhanced Feature Engineering

```python
# src/data_processing/feature_engineering.py
class FeatureEngineer:
    """Enhanced feature engineering with configurable indicators"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.indicators = self._load_indicators()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features with configuration-based selection"""

    def _load_indicators(self) -> Dict[str, Indicator]:
        """Load indicators based on configuration"""

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
```

### 2.3 Streaming Data Processor

```python
# src/data_processing/streaming_processor.py
class StreamingDataProcessor:
    """Enhanced streaming processor for large datasets"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.memory_limit = config.memory_limit

    async def process_stream(
        self,
        csv_path: Path,
        feature_engineer: FeatureEngineer
    ) -> pd.DataFrame:
        """Process large CSV files in streaming fashion"""

    def _validate_csv_format(self, csv_path: Path) -> CSVFormat:
        """Validate and detect CSV format"""

    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        feature_engineer: FeatureEngineer
    ) -> pd.DataFrame:
        """Process individual chunk with feature engineering"""
```

---

## 3. Implementation Steps

### Step 1: Create Enhanced Pipeline Interface
- [ ] Create `src/pipelines/__init__.py`
- [ ] Create `src/pipelines/hmm_pipeline.py`
- [ ] Create `src/pipelines/pipeline_types.py`
- [ ] Implement `HMMPipeline` class
- [ ] Add async support for large datasets
- [ ] Add progress tracking and logging

### Step 2: Enhance Feature Engineering
- [ ] Extend `src/data_processing/feature_engineering.py`
- [ ] Add configurable indicator selection
- [ ] Add custom indicator support
- [ ] Add feature validation and quality checks
- [ ] Add feature importance analysis

### Step 3: Create Streaming Processor
- [ ] Create `src/data_processing/streaming_processor.py`
- [ ] Add memory optimization features
- [ ] Add progress tracking
- [ ] Add error recovery and chunk retry logic
- [ ] Add parallel processing support

### Step 4: Enhance Model Training Integration
- [ ] Update `src/model_training/hmm_trainer.py`
- [ ] Add pipeline integration methods
- [ ] Add model selection and comparison
- [ ] Add hyperparameter optimization
- [ ] Add training progress tracking

### Step 5: Create Unified CLI Interface
- [ ] Create `src/cli/hmm_commands.py`
- [ ] Add subcommand structure
- [ ] Add configuration file support
- [ ] Add progress bars and status indicators
- [ ] Add batch processing support

### Step 6: Add Configuration Integration
- [ ] Update `src/utils/config.py`
- [ ] Add pipeline configuration sections
- [ ] Add feature engineering configurations
- [ ] Add streaming configurations
- [ ] Add CLI configuration integration

### Step 7: Create Backward Compatibility Layer
- [ ] Create `src/compatibility/main_adapter.py`
- [ ] Add argument translation
- [ ] Maintain exact output formats
- [ ] Add deprecation warnings
- [ ] Create migration guide

---

## 4. Code Quality Enhancements

### 4.1 Type Safety
```python
from typing import (
    Optional, List, Dict, Any, Union,
    Callable, AsyncIterator, Tuple
)
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration for HMM pipeline"""
    features: FeatureConfig
    training: TrainingConfig
    persistence: PersistenceConfig
    streaming: StreamingConfig
    backtesting: Optional[BacktestConfig] = None
```

### 4.2 Error Handling
```python
class PipelineError(Exception):
    """Base pipeline exception"""
    pass

class FeatureEngineeringError(PipelineError):
    """Feature engineering specific errors"""
    pass

class ModelTrainingError(PipelineError):
    """Model training specific errors"""
    pass

class DataProcessingError(PipelineError):
    """Data processing specific errors"""
    pass
```

### 4.3 Logging and Monitoring
```python
import structlog
from tqdm import tqdm
from time import perf_counter

logger = structlog.get_logger()

class PipelineProgress:
    """Pipeline progress tracking"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = perf_counter()

    def step(self, description: str) -> None:
        """Advance to next step"""
        self.current_step += 1
        elapsed = perf_counter() - self.start_time
        logger.info(
            "Pipeline progress",
            step=self.current_step,
            total=self.total_steps,
            description=description,
            elapsed=elapsed
        )
```

---

## 5. Testing Strategy

### 5.1 Unit Tests
```python
# tests/pipelines/test_hmm_pipeline.py
class TestHMMPipeline:
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""

    def test_feature_engineering_integration(self):
        """Test feature engineering integration"""

    def test_model_training_integration(self):
        """Test model training integration"""

    @pytest.mark.asyncio
    async def test_streaming_processing(self):
        """Test streaming data processing"""
```

### 5.2 Integration Tests
```python
# tests/integration/test_main_migration.py
class TestMainMigration:
    def test_backward_compatibility(self):
        """Test backward compatibility with main.py"""

    def test_output_format_consistency(self):
        """Test output format consistency"""

    def test_large_dataset_processing(self):
        """Test processing of large datasets"""
```

---

## 6. Migration Validation

### 6.1 Output Comparison Tests
```python
def compare_outputs(
    original_output: Path,
    migrated_output: Path,
    tolerance: float = 1e-10
) -> bool:
    """Compare original and migrated outputs"""

    # Load both outputs
    original = pd.read_csv(original_output)
    migrated = pd.read_csv(migrated_output)

    # Compare structure
    assert original.columns.tolist() == migrated.columns.tolist()
    assert len(original) == len(migrated)

    # Compare values (within tolerance)
    for col in original.columns:
        if pd.api.types.is_numeric_dtype(original[col]):
            np.testing.assert_allclose(
                original[col].values,
                migrated[col].values,
                rtol=tolerance,
                equal_nan=True
            )
        else:
            assert original[col].equals(migrated[col])

    return True
```

### 6.2 Performance Benchmarks
```python
def benchmark_performance(
    csv_path: Path,
    original_runtime: float,
    target_improvement: float = 0.10
) -> Dict[str, float]:
    """Benchmark migrated performance against original"""

    start_time = perf_counter()

    # Run migrated pipeline
    pipeline = HMMPipeline.from_config("config/default.yaml")
    result = await pipeline.run(csv_path)

    runtime = perf_counter() - start_time
    improvement = (original_runtime - runtime) / original_runtime

    return {
        "original_runtime": original_runtime,
        "migrated_runtime": runtime,
        "improvement_ratio": improvement,
        "target_met": improvement >= target_improvement
    }
```

---

## 7. Rollout Strategy

### 7.1 Phase 1: Core Migration (Week 1)
- Implement basic pipeline structure
- Migrate feature engineering
- Create streaming processor
- Basic testing and validation

### 7.2 Phase 2: Enhancement (Week 2)
- Add advanced features
- Implement CLI enhancements
- Add configuration integration
- Comprehensive testing

### 7.3 Phase 3: Validation (Week 3)
- Performance optimization
- Backward compatibility validation
- Documentation updates
- Final testing and sign-off

---

## 8. Success Criteria

### 8.1 Functional Requirements
- [ ] All main.py functionality replicated
- [ ] Output format consistency maintained
- [ ] Backward compatibility preserved
- [ ] Error handling improved

### 8.2 Performance Requirements
- [ ] Processing speed improved by ≥10%
- [ ] Memory usage optimized by ≥15%
- [ ] Better progress tracking and logging
- [ ] Enhanced error recovery

### 8.3 Quality Requirements
- [ ] 95%+ test coverage
- [ ] Type safety with mypy validation
- [ ] Comprehensive documentation
- [ ] CLI help and examples

---

## 9. Risk Mitigation

### 9.1 Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Output format changes | High | Medium | Comprehensive comparison tests |
| Performance regression | High | Low | Performance benchmarks and optimization |
| Memory issues | Medium | Low | Streaming processing and memory limits |
| Integration failures | Medium | Low | Incremental migration and testing |

### 9.2 Migration Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Lost functionality | High | Low | Feature mapping and validation |
| Breaking changes | High | Medium | Backward compatibility layer |
| Extended timeline | Medium | Medium | Incremental delivery and parallel work |
| Quality issues | Medium | Low | Comprehensive testing and code review |

---

## 10. Deliverables

### 10.1 Code Deliverables
- [ ] Enhanced `src/pipelines/hmm_pipeline.py`
- [ ] Updated `src/data_processing/feature_engineering.py`
- [ ] New `src/data_processing/streaming_processor.py`
- [ ] Enhanced `src/cli/hmm_commands.py`
- [ ] Updated configuration system
- [ ] Backward compatibility layer

### 10.2 Documentation Deliverables
- [ ] API documentation for new modules
- [ ] Migration guide for users
- [ ] Configuration examples
- [ ] CLI usage guide

### 10.3 Testing Deliverables
- [ ] Unit test suite
- [ ] Integration test suite
- [ ] Performance benchmarks
- [ ] Validation test suite

---

## Implementation Status

**Current Phase**: Planning and Design ✅
**Next Phase**: Implementation (Step 1: Create Pipeline Interface)

**Estimated Timeline**: 15-20 hours
**Expected Completion**: End of Week 1

---

## Notes and Decisions

1. **Async Processing**: Using async/await for better I/O handling in streaming operations
2. **Configuration-Driven**: All features and parameters configurable via YAML files
3. **Backward Compatibility**: Maintaining exact CLI compatibility for smooth transition
4. **Progress Tracking**: Enhanced logging and progress bars for better UX
5. **Memory Optimization**: Streaming processing with configurable memory limits
6. **Type Safety**: Full type annotations with mypy validation
7. **Error Recovery**: Robust error handling with recovery mechanisms

This migration provides a solid foundation for future enhancements while maintaining compatibility with existing workflows.