# Src Directory Structure Design - Enhanced Architecture

## Executive Summary

**Current State**: Well-organized src directory with 8 main modules and 33 files
**Proposed Enhancements**: Add 4 new modules, restructure existing modules, enhance integration patterns
**Total Files After Migration**: ~60 files (including main.py functionality migration)
**Complexity**: Medium to High (modular with clear separation of concerns)
**Migration Priority**: High (Foundation for all subsequent migration phases)

Based on the comprehensive analysis of main directory files, this document proposes enhancements to the src directory structure to accommodate all main directory functionality while maintaining modularity, extensibility, and professional architecture patterns.

---

## Current Src Directory Analysis

### Existing Structure Assessment

```
src/
├── backtesting/           # 6 files - Strategy engine, performance analysis
├── data_processing/       # 4 files - CSV parsing, validation, feature engineering
├── hmm_models/           # 5 files - HMM models, factory pattern
├── model_training/       # 4 files - Training, inference, persistence
├── processing_engines/   # 7 files - Streaming, Dask, Daft engines
├── utils/                # 4 files - Configuration, logging, data types
└── visualization/        # 4 files - Charts, dashboards, reports
```

**Strengths of Current Structure**:
- **Modular Design**: Clear separation of concerns
- **Factory Patterns**: Dynamic model and engine creation
- **Comprehensive Coverage**: All major functionality areas represented
- **Professional Architecture**: Well-organized and maintainable

**Gaps Identified**:
- **CLI Interface**: No dedicated CLI module
- **Deep Learning**: No LSTM/neural network support
- **Algorithm Comparison**: No framework for comparing different algorithms
- **Configuration Management**: Limited configuration consolidation
- **Testing Infrastructure**: No dedicated testing framework
- **Advanced Analytics**: Limited algorithm selection and optimization

---

## Enhanced Src Directory Structure

### Proposed New Structure

```
src/
├── algorithms/              # NEW: Algorithm management and comparison
│   ├── __init__.py
│   ├── base.py             # Base algorithm interface
│   ├── hmm_algorithm.py    # HMM algorithm wrapper
│   ├── lstm_algorithm.py   # LSTM algorithm wrapper
│   ├── hybrid_algorithm.py # Hybrid HMM-LSTM algorithm
│   ├── factory.py          # Algorithm factory pattern
│   ├── selection.py        # Automatic algorithm selection
│   └── comparison.py       # Algorithm comparison framework
├── backtesting/            # ENHANCED: Advanced backtesting capabilities
│   ├── __init__.py
│   ├── strategy_engine.py  # Existing - strategy framework
│   ├── performance_analyzer.py # Existing - performance metrics
│   ├── bias_prevention.py  # Existing - lookahead bias prevention
│   ├── portfolio_manager.py # NEW: Multi-asset portfolio management
│   ├── risk_manager.py     # NEW: Risk management tools
│   ├── transaction_costs.py # NEW: Realistic cost modeling
│   ├── performance_metrics.py # Existing - detailed metrics
│   └── utils.py            # Existing - backtesting utilities
├── cli/                    # NEW: Unified CLI framework
│   ├── __init__.py
│   ├── main.py             # Main CLI interface
│   ├── commands/           # CLI command implementations
│   │   ├── __init__.py
│   │   ├── analyze.py      # Analysis command
│   │   ├── validate.py     # Data validation command
│   │   ├── infer.py        # Model inference command
│   │   ├── compare.py      # Algorithm comparison command
│   │   └── version.py      # Version information command
│   ├── config.py           # CLI configuration management
│   ├── progress.py         # Progress tracking and monitoring
│   └── output.py           # Output formatting and reporting
├── configuration/          # NEW: Centralized configuration management
│   ├── __init__.py
│   ├── manager.py          # Configuration manager
│   ├── schemas.py          # Configuration schemas and validation
│   ├── loaders.py          # Configuration file loaders (YAML, JSON)
│   ├── defaults.py         # Default configuration values
│   └── validators.py       # Configuration validation utilities
├── data_processing/        # ENHANCED: Advanced data processing
│   ├── __init__.py
│   ├── csv_parser.py       # Existing - CSV parsing with format detection
│   ├── data_validation.py  # Existing - data quality validation
│   ├── feature_engineering.py # Existing - technical indicators
│   ├── data_cleaner.py     # NEW: Advanced data cleaning
│   ├── resampler.py        # NEW: Time series resampling
│   ├── quality_metrics.py  # NEW: Data quality assessment
│   └── formatters.py       # NEW: Data format conversion
├── deep_learning/          # NEW: Deep learning algorithms
│   ├── __init__.py
│   ├── lstm_model.py       # LSTM neural network implementation
│   ├── transformer_model.py # NEW: Transformer-based models
│   ├── neural_utils.py     # Neural network utilities
│   ├── training.py         # Deep learning training framework
│   ├── inference.py        # Model inference utilities
│   └── preprocessing.py    # Deep learning data preprocessing
├── hmm_models/             # ENHANCED: Advanced HMM implementations
│   ├── __init__.py
│   ├── base.py             # Existing - base HMM interface
│   ├── gaussian_hmm.py     # Existing - Gaussian HMM implementation
│   ├── gmm_hmm.py          # Existing - GMM HMM implementation
│   ├── factory.py          # Existing - HMM factory pattern
│   ├── selection.py        # NEW: HMM model selection
│   ├── ensemble.py         # NEW: HMM ensemble methods
│   ├── online_hmm.py       # NEW: Online learning HMM
│   └── custom_hmm.py       # NEW: Custom HMM implementations
├── model_training/         # ENHANCED: Advanced model training
│   ├── __init__.py
│   ├── hmm_trainer.py      # Existing - HMM training
│   ├── inference_engine.py # Existing - State inference
│   ├── model_persistence.py # Existing - Model save/load
│   ├── hyperparameter_tuning.py # NEW: Hyperparameter optimization
│   ├── cross_validation.py # NEW: Cross-validation framework
│   ├── model_registry.py   # NEW: Model registry and metadata
│   ├── training_monitor.py # NEW: Training progress monitoring
│   └── ensemble_training.py # NEW: Ensemble model training
├── monitoring/             # NEW: System monitoring and metrics
│   ├── __init__.py
│   ├── performance_monitor.py # Performance tracking
│   ├── memory_monitor.py   # Memory usage monitoring
│   ├── resource_monitor.py # System resource monitoring
│   ├── metrics_collector.py # Metrics collection and storage
│   ├── alerts.py           # Alerting system
│   └── dashboard.py        # Monitoring dashboard
├── processing_engines/     # ENHANCED: Advanced processing engines
│   ├── __init__.py
│   ├── streaming_engine.py # Existing - Streaming processing
│   ├── dask_engine.py      # Existing - Dask distributed processing
│   ├── daft_engine.py      # Existing - Daft out-of-core processing
│   ├── factory.py          # Existing - Engine factory
│   ├── index.py            # Existing - Engine registry
│   ├── batch_engine.py     # NEW: Batch processing engine
│   ├── realtime_engine.py  # NEW: Real-time processing engine
│   └── optimization.py     # NEW: Engine optimization utilities
├── testing/                # NEW: Comprehensive testing framework
│   ├── __init__.py
│   ├── unit/               # Unit tests
│   │   ├── __init__.py
│   │   ├── test_data_processing.py
│   │   ├── test_hmm_models.py
│   │   ├── test_backtesting.py
│   │   └── test_algorithms.py
│   ├── integration/        # Integration tests
│   │   ├── __init__.py
│   │   ├── test_pipelines.py
│   │   ├── test_engines.py
│   │   └── test_cli.py
│   ├── performance/        # Performance tests
│   │   ├── __init__.py
│   │   ├── test_benchmarks.py
│   │   ├── test_memory.py
│   │   └── test_scalability.py
│   ├── fixtures/           # Test data and fixtures
│   │   ├── __init__.py
│   │   ├── sample_data.py
│   │   └── mock_models.py
│   └── utils.py            # Testing utilities
├── utils/                  # ENHANCED: Advanced utilities
│   ├── __init__.py
│   ├── config.py           # Existing - Configuration utilities
│   ├── logging_config.py   # Existing - Logging configuration
│   ├── data_types.py       # Existing - Data type definitions
│   ├── decorators.py       # NEW: Function decorators
│   ├── validators.py       # NEW: Data validation utilities
│   ├── serializers.py      # NEW: Data serialization utilities
│   ├── math_utils.py       # NEW: Mathematical utilities
│   └── time_utils.py       # NEW: Time series utilities
└── visualization/          # ENHANCED: Advanced visualization
    ├── __init__.py
    ├── chart_generator.py  # Existing - Chart generation
    ├── dashboard_builder.py # Existing - Dashboard creation
    ├── report_generator.py # Existing - Report generation
    ├── interactive_plots.py # NEW: Interactive plotting
    ├── animations.py       # NEW: Plot animations
    ├── themes.py           # NEW: Plotting themes and styles
    └── exporters.py        # NEW: Plot export utilities
```

---

## Module Enhancement Details

### 1. algorithms/ - Algorithm Management Framework

**Purpose**: Unified interface for different algorithm types with automatic selection and comparison

**Key Components**:
- **Base Algorithm Interface**: Standardized API across all algorithms
- **Algorithm Factory**: Dynamic algorithm creation and configuration
- **Selection Framework**: Automatic algorithm selection based on data characteristics
- **Comparison Tools**: Side-by-side algorithm performance comparison

**Integration with Main Directory**:
- **LSTM.py**: Migrated as `lstm_algorithm.py`
- **main.py HMM**: Enhanced as `hmm_algorithm.py`
- **Hybrid Models**: New `hybrid_algorithm.py` for combined approaches

**Example Usage**:
```python
from algorithms import AlgorithmFactory, AlgorithmSelector

# Automatic algorithm selection
selector = AlgorithmSelector()
best_algorithm = selector.select_algorithm(data, target='regime_detection')

# Manual algorithm creation
factory = AlgorithmFactory()
hmm_algo = factory.create_algorithm('hmm', n_states=3)
lstm_algo = factory.create_algorithm('lstm', window_size=60)

# Algorithm comparison
comparison = factory.compare_algorithms(data, ['hmm', 'lstm', 'hybrid'])
```

### 2. cli/ - Unified Command-Line Interface

**Purpose**: Consolidate all CLI functionality from main directory into unified framework

**Key Components**:
- **Command Organization**: Hierarchical command structure with subcommands
- **Mode Selection**: Simple, standard, and advanced operation modes
- **Progress Tracking**: Real-time progress monitoring across all operations
- **Configuration Integration**: Seamless integration with configuration management

**Main Directory Integration**:
- **cli.py**: Migrated as comprehensive mode
- **cli_simple.py**: Migrated as simple mode
- **cli_comprehensive.py**: Features integrated into advanced mode

**Command Structure**:
```python
# Unified CLI with mode selection
hmm-analysis --mode simple analyze -i data.csv
hmm-analysis --mode standard analyze -i data.csv --n-states 4
hmm-analysis --mode advanced analyze -i data.csv --engine daft --memory-monitor

# Specialized commands
hmm-analysis validate -i data.csv
hmm-analysis infer -m model.pkl -i new_data.csv
hmm-analysis compare algorithms -i data.csv --models hmm,lstm,hybrid
```

### 3. configuration/ - Centralized Configuration Management

**Purpose**: Unified configuration system supporting multiple sources and formats

**Key Components**:
- **Configuration Manager**: Centralized configuration loading and validation
- **Schema Validation**: Pydantic-based configuration validation
- **Multiple Sources**: CLI arguments, config files, environment variables
- **Dynamic Updates**: Runtime configuration updates

**Configuration Hierarchy**:
```python
# Configuration sources (in priority order):
1. CLI arguments (highest priority)
2. Environment variables
3. Configuration files (YAML/JSON)
4. Default values (lowest priority)

# Example configuration file
analysis:
  n_states: 3
  covariance_type: "full"
  random_seed: 42

processing:
  engine: "streaming"
  chunk_size: 100000

features:
  indicators:
    - name: "atr"
      window: 14
    - name: "rsi"
      window: 14
```

### 4. deep_learning/ - Deep Learning Framework

**Purpose**: Comprehensive deep learning capabilities beyond LSTM

**Key Components**:
- **LSTM Implementation**: Enhanced version of main.py LSTM
- **Transformer Models**: Advanced sequence modeling
- **Training Framework**: Optimized training loops with monitoring
- **Preprocessing Pipeline**: Deep learning-specific data preparation

**Enhanced LSTM Features**:
```python
from deep_learning import LSTMModel, TrainingConfig

# Enhanced LSTM with configuration
config = TrainingConfig(
    epochs=100,
    batch_size=32,
    early_stopping=True,
    validation_split=0.2,
    callbacks=['tensorboard', 'checkpoint']
)

model = LSTMModel(
    sequence_length=60,
    hidden_units=[64, 64],
    dropout_rate=0.5,
    optimizer='adam'
)
```

### 5. monitoring/ - System Monitoring and Metrics

**Purpose**: Production-ready monitoring and performance tracking

**Key Components**:
- **Performance Monitor**: Real-time performance metrics
- **Memory Monitor**: Memory usage tracking and optimization
- **Alert System**: Automatic alerting for performance issues
- **Metrics Collection**: Structured metrics storage and analysis

**Monitoring Features**:
```python
from monitoring import PerformanceMonitor, MemoryMonitor

# Performance monitoring
monitor = PerformanceMonitor()
with monitor.track_operation("data_processing"):
    result = process_large_dataset(data)

# Memory monitoring
memory_monitor = MemoryMonitor()
if memory_monitor.memory_usage > 0.8:
    memory_monitor.trigger_cleanup()
```

### 6. testing/ - Comprehensive Testing Framework

**Purpose**: Professional testing infrastructure with multiple test types

**Key Components**:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmarking and scalability testing
- **Test Fixtures**: Reusable test data and utilities

**Testing Structure**:
```python
# Unit tests
from testing.unit import test_hmm_models, test_feature_engineering

# Integration tests
from testing.integration import test_full_pipeline, test_cli_workflows

# Performance tests
from testing.performance import test_processing_speed, test_memory_usage

# Test fixtures
from testing.fixtures import sample_ohlcv_data, mock_hmm_model
```

---

## Migration Path for Main Directory Functionality

### 1. main.py Integration

**Current main.py Functions → Enhanced src Structure**:

```python
# Current main.py structure
def add_features(df: pd.DataFrame) -> pd.DataFrame
def stream_features(csv_path: Path, chunksize: int) -> pd.DataFrame
def simple_backtest(df: pd.DataFrame, states: np.ndarray) -> pd.Series
def perf_metrics(series: pd.Series) -> tuple
def main(args)

# Enhanced src structure integration
from data_processing import DataProcessor, FeatureEngineer
from algorithms import HMMAlgorithm
from backtesting import BacktestEngine, PerformanceAnalyzer
from cli import main as cli_main
```

**Migration Strategy**:
1. **Feature Engineering**: Enhance `data_processing/feature_engineering.py`
2. **Streaming Processing**: Enhance `processing_engines/streaming_engine.py`
3. **Backtesting**: Enhance `backtesting/strategy_engine.py`
4. **CLI Integration**: Create unified CLI in `cli/`

### 2. CLI Implementations Integration

**Current CLI Files → Unified CLI Framework**:

```python
# Current: Three separate CLI files
cli.py              # Comprehensive CLI
cli_simple.py       # Simple CLI
cli_comprehensive.py # Advanced CLI

# Enhanced: Single unified CLI with modes
from cli import main
# Usage: hmm-analysis --mode [simple|standard|advanced] <command>
```

**Migration Benefits**:
- **Code Consolidation**: Eliminate duplicate functionality
- **Feature Unification**: Best features from all implementations
- **Maintenance Reduction**: Single codebase to maintain
- **User Experience**: Consistent interface across all modes

### 3. Specialized Scripts Integration

**Current Specialized Scripts → Enhanced src Structure**:

```python
# Current: Standalone specialized scripts
LSTM.py                 # LSTM implementation
hmm_futures_daft.py    # Daft processing engine
hmm_futures_script.py  # Dask-based HMM

# Enhanced: Integrated into src structure
from algorithms import LSTMAlgorithm, HybridAlgorithm
from processing_engines import DaftEngine, DaskEngine
from deep_learning import LSTMModel, TrainingFramework
```

---

## Enhanced Architecture Patterns

### 1. Factory Pattern Enhancement

**Algorithm Factory**:
```python
class AlgorithmFactory:
    _algorithms = {
        'hmm': HMMAlgorithm,
        'lstm': LSTMAlgorithm,
        'hybrid': HybridAlgorithm,
        'transformer': TransformerAlgorithm
    }

    @classmethod
    def create_algorithm(cls, algorithm_type: str, **kwargs):
        if algorithm_type not in cls._algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        return cls._algorithms[algorithm_type](**kwargs)

    @classmethod
    def list_algorithms(cls):
        return list(cls._algorithms.keys())
```

**Processing Engine Factory**:
```python
class ProcessingEngineFactory:
    _engines = {
        'streaming': StreamingEngine,
        'dask': DaskEngine,
        'daft': DaftEngine,
        'batch': BatchEngine,
        'realtime': RealtimeEngine
    }

    @classmethod
    def create_engine(cls, engine_type: str, **kwargs):
        engine_class = cls._engines.get(engine_type)
        if not engine_class:
            raise ValueError(f"Unknown engine type: {engine_type}")
        return engine_class(**kwargs)
```

### 2. Configuration Management Pattern

**Centralized Configuration**:
```python
class ConfigurationManager:
    def __init__(self):
        self.config = {}
        self.load_default_config()

    def load_config(self, sources: List[str]):
        for source in sources:
            if source.endswith('.yaml'):
                self.config.update(self._load_yaml(source))
            elif source.endswith('.json'):
                self.config.update(self._load_json(source))

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def validate(self):
        return ConfigValidator.validate(self.config)
```

### 3. Plugin Architecture Pattern

**Algorithm Plugin System**:
```python
class AlgorithmPlugin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AlgorithmRegistry.register(cls)

class AlgorithmRegistry:
    _algorithms = {}

    @classmethod
    def register(cls, algorithm_class):
        cls._algorithms[algorithm_class.name] = algorithm_class

    @classmethod
    def get_algorithm(cls, name: str):
        return cls._algorithms.get(name)
```

---

## Performance and Scalability Enhancements

### 1. Memory Optimization

**Memory Monitoring Integration**:
```python
class MemoryOptimizedProcessor:
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold
        self.memory_monitor = MemoryMonitor()

    def process_with_monitoring(self, data):
        with self.memory_monitor.track():
            result = self.process(data)
            if self.memory_monitor.usage > self.memory_threshold:
                self.optimize_memory_usage()
        return result
```

### 2. Parallel Processing

**Enhanced Processing Engines**:
```python
class ParallelProcessingEngine:
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or os.cpu_count()

    def process_parallel(self, data_chunks: List[pd.DataFrame]):
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk)
                      for chunk in data_chunks]
            results = [future.result() for future in futures]
        return pd.concat(results, ignore_index=True)
```

### 3. Caching Strategy

**Intelligent Caching**:
```python
class CachedProcessor:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def process_with_cache(self, data_hash: str, processing_func):
        cache_file = self.cache_dir / f"{data_hash}.pkl"
        if cache_file.exists():
            return joblib.load(cache_file)

        result = processing_func()
        joblib.dump(result, cache_file)
        return result
```

---

## Integration Testing Strategy

### 1. Module Integration Tests

**Cross-Module Testing**:
```python
def test_algorithm_with_processing_engine():
    # Test algorithm with different processing engines
    for engine_type in ['streaming', 'dask', 'daft']:
        engine = ProcessingEngineFactory.create_engine(engine_type)
        algorithm = AlgorithmFactory.create_algorithm('hmm')

        data = engine.load_data('test_data.csv')
        result = algorithm.train(data)
        assert result is not None

def test_cli_with_configuration():
    # Test CLI with different configuration sources
    for config_source in ['config.yaml', 'config.json', 'env_vars']:
        config = ConfigurationManager()
        config.load_config([config_source])

        result = run_cli_with_config(config)
        assert result.success
```

### 2. Performance Integration Tests

**End-to-End Performance**:
```python
def test_full_pipeline_performance():
    start_time = time.time()

    # Test complete pipeline
    data = load_test_data()
    features = engineer_features(data)
    model = train_hmm_model(features)
    results = backtest_strategy(model, data)

    execution_time = time.time() - start_time
    assert execution_time < MAX_EXECUTION_TIME
    assert len(results.trades) > MIN_TRADES
```

---

## Documentation and Developer Experience

### 1. API Documentation

**Comprehensive API Docs**:
```python
class HMMAlgorithm:
    """
    Hidden Markov Model algorithm for regime detection.

    Args:
        n_states: Number of hidden states
        covariance_type: Covariance matrix type
        random_state: Random seed for reproducibility

    Example:
        >>> algorithm = HMMAlgorithm(n_states=3)
        >>> algorithm.fit(data)
        >>> states = algorithm.predict(data)
    """
```

### 2. Configuration Documentation

**Configuration Reference**:
```yaml
# Example configuration with documentation
analysis:
  # Number of HMM states to detect (2-10 recommended)
  n_states: 3

  # Covariance type: 'full', 'diag', 'spherical', 'tied'
  covariance_type: "full"

processing:
  # Processing engine: 'streaming', 'dask', 'daft'
  engine: "streaming"

  # Chunk size for processing (rows)
  chunk_size: 100000
```

### 3. Developer Guides

**Getting Started Guide**:
```markdown
# HMM Analysis System - Developer Guide

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run simple analysis: `hmm-analysis --mode simple analyze -i data.csv`
3. Run advanced analysis: `hmm-analysis --mode advanced analyze -i data.csv --engine daft`

## Adding New Algorithms
1. Inherit from `AlgorithmBase` class
2. Implement required methods
3. Register in `AlgorithmFactory`
4. Add tests
```

---

## Migration Implementation Plan

### Phase 1: Core Structure Enhancement
1. **Create New Modules**: Set up empty module structures
2. **Define Interfaces**: Create base classes and interfaces
3. **Implement Factories**: Set up factory patterns
4. **Configuration System**: Implement centralized configuration

### Phase 2: Main Directory Integration
1. **Feature Engineering**: Enhance data processing with main.py features
2. **CLI Unification**: Consolidate CLI implementations
3. **Algorithm Integration**: Migrate specialized scripts
4. **Backtesting Enhancement**: Integrate main.py backtesting

### Phase 3: Advanced Features
1. **Deep Learning**: Implement enhanced LSTM framework
2. **Monitoring**: Add system monitoring capabilities
3. **Testing Framework**: Implement comprehensive testing
4. **Performance Optimization**: Add caching and optimization

### Phase 4: Documentation and Validation
1. **API Documentation**: Generate comprehensive API docs
2. **User Guides**: Create user documentation
3. **Developer Docs**: Write developer guides
4. **Integration Testing**: Comprehensive system validation

---

## Success Criteria

### Functional Requirements
- [ ] All main directory functionality successfully migrated
- [ ] Enhanced features implemented and tested
- [ ] Unified CLI interface operational
- [ ] Configuration management working
- [ ] Performance improvements validated

### Technical Requirements
- [ ] Modular architecture maintained
- [ ] Factory patterns implemented
- [ ] Plugin system working
- [ ] Memory optimization effective
- [ ] Testing coverage >90%

### Quality Requirements
- [ ] Code follows Python best practices
- [ ] Comprehensive documentation
- [ ] Performance benchmarks met
- [ ] User experience improved
- [ ] Maintainability enhanced

---

## Conclusion

The enhanced src directory structure provides a professional, modular, and extensible architecture that can accommodate all main directory functionality while adding significant new capabilities. The design emphasizes:

- **Modularity**: Clear separation of concerns with well-defined interfaces
- **Extensibility**: Plugin architecture for easy addition of new algorithms
- **Performance**: Memory optimization and parallel processing capabilities
- **Maintainability**: Comprehensive testing and documentation
- **User Experience**: Unified CLI with multiple operation modes

This architecture will serve as a solid foundation for the migration and future development of the HMM analysis system, enabling it to scale from simple analysis tasks to enterprise-grade financial modeling applications.

---

*Structure Design Completed: October 23, 2025*
*Next Step: Phase 1.2.2 - Create migration strategy*