# Main Directory Function and Class Analysis

## Subtask 1.1.1.2: Analyze Each Python File Structure

### Function and Class Definition Analysis

---

## Core HMM Analysis Scripts

### main.py

#### Function Definitions
```python
def add_features(df: pd.DataFrame) -> pd.DataFrame
def stream_features(csv_path: Path, chunksize: int = 100_000) -> pd.DataFrame
def simple_backtest(df: pd.DataFrame, states: np.ndarray) -> pd.Series
def perf_metrics(series: pd.Series) -> tuple
def main(args)
```

#### Class Definitions
- No class definitions (functional programming style)
- All functions operate in global scope
- Direct imports from external libraries

#### Key Analysis Points
- **Function Count**: 5 main functions
- **Architecture**: Functional programming approach
- **Scope**: Global function definitions
- **Dependencies**: Direct imports from sklearn, numpy, pandas, hmmlearn

---

### CLI Interface Scripts

### cli.py

#### Function Definitions
```python
def cli()
def analyze()
def validate()
def version()
```

#### Class Definitions
- No class definitions
- Uses Click decorators for CLI structure
- Command group organization

#### Key Analysis Points
- **Function Count**: 4 main functions + CLI decorators
- **Architecture**: Click-based CLI framework
- **Command Structure**: Hierarchical command organization
- **Dependencies**: Click, Dash, Pandas, NumPy, sys, pathlib

### cli_simple.py

#### Function Definitions
```python
def main()
```

#### Class Definitions
- No class definitions
- Simple CLI implementation
- Basic command structure

#### Key Analysis Points
- **Function Count**: 1 main function
- **Architecture**: Simple CLI implementation
- **Command Structure**: Basic argument parsing
- **Dependencies**: Click, sys

### cli_comprehensive.py

#### Function Definitions
```python
def cli()
def analyze()
def validate()
def version()
```

#### Class Definitions
- No class definitions
- Advanced CLI with rich features
- Comprehensive command structure

#### Key Analysis Points
- **Function Count**: 4 main functions + CLI decorators
- **Architecture**: Advanced Click-based CLI
- **Command Structure**: Rich feature set with subcommands
- **Dependencies**: Click, Dash, Pandas, NumPy

---

## Algorithm Implementation Scripts

### LSTM.py

#### Function Definitions
```python
def create_sequences(data, lookback=60):
def train_model(X_train, y_train, input_size, hidden_size, num_epochs, batch_size):
def evaluate_model(model, X_test, y_test):
def plot_results(history, predictions, actual):
def main()
```

#### Class Definitions
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
    def forward(self, x):
```

#### Key Analysis Points
- **Function Count**: 6 functions + 1 class
- **Architecture**: PyTorch neural network implementation
- **Scope**: Deep learning with LSTM
- **Dependencies**: PyTorch, NumPy, Pandas, sklearn
- **Integration**: Potential for hybrid HMM-LSTM models

### hmm_futures_daft.py

#### Function Definitions
```python
def main()
```

#### Class Definitions
- No class definitions
- Script-based implementation
- Daft engine integration

#### Key Analysis Points
- **Function Count**: 1 main function
- **Architecture**: Script-based with Daft integration
- **Scope**: Distributed HMM processing
- **Dependencies**: Daft, Pandas, NumPy
- **Integration**: Enhanced processing engine capability

### hmm_futures_script.py

#### Function Definitions
```python
def load_data(file_path):
def preprocess_data(data):
def train_hmm_model(features, n_states=3):
def predict_states(model, features):
def backtest_strategy(states, prices):
def plot_results(data, states, predictions)
def main()
```

#### Class Definitions
- No class definitions
- Complete pipeline implementation
- End-to-end workflow

#### Key Analysis Points
- **Function Count**: 7 functions
- **Architecture**: Complete pipeline implementation
- **Scope**: Full HMM workflow from data to visualization
- **Dependencies**: NumPy, Pandas, sklearn
- **Integration**: Self-contained analysis pipeline

---

## Testing Scripts - Core Functionality

### test_main.py

#### Function Definitions
```python
def test_data_loading()
def test_feature_engineering()
def test_hmm_training()
def test_state_prediction()
def test_backtesting()
def main()
```

#### Class Definitions
- No class definitions
- Basic testing framework
- Assertion-based validation

#### Key Analysis Points
- **Function Count**: 5 test functions + 1 main
- **Architecture**: Simple test framework
- **Scope**: Main.py functionality validation
- **Dependencies**: NumPy, Pandas

### test_hmm_models.py

#### Function Definitions
```python
def generate_test_data(n_samples=1000, n_features=5, random_state=42)
def test_hmm_models()
def main()
```

#### Class Definitions
- No class definitions
- Comprehensive HMM model testing
- Factory pattern testing

#### Key Analysis Points
- **Function Count**: 3 functions + 1 main
- **Architecture**: HMM model testing framework
- **Scope**: HMM model functionality validation
- **Dependencies**: NumPy, Pandas, sys

### test_hmm_training.py

#### Function Definitions
```python
def generate_sample_data(n_samples=1000, n_features=10):
def test_hmm_training_pipeline()
def main()
```

#### Class Definitions
- No class definitions
- Training pipeline testing
- End-to-end validation

#### Key Analysis Points
- **Function Count**: 3 functions + 1 main
- **Architecture**: Training pipeline testing
- **Scope**: HMM training process validation
- **Dependencies**: NumPy, Pandas, sklearn

---

## Testing Scripts - CLI Interface

### test_cli.py

#### Function Definitions
```python
def test_cli_help()
def test_cli_analyze()
def test_cli_validate()
def test_cli_version()
def main()
```

#### Class Definitions
- No class definitions
- CLI functionality testing
- Command validation

#### Key Analysis Points
- **Function Count**: 5 functions + 1 main
- **Architecture**: CLI testing framework
- **Scope**: CLI command validation
- **Dependencies**: Click, subprocess

### test_cli_simple.py

#### Function Definitions
```python
def test_cli_simple_help()
def test_cli_simple_functionality()
def main()
```

#### Class Definitions
- No class definitions
- Simple CLI testing
- Basic validation

#### Key Analysis Points
- **Function Count**: 3 functions + 1 main
- **Architecture**: Simple CLI testing
- **Scope**: Basic CLI functionality
- **Dependencies**: Click

### test_core_cli.py

#### Function Definitions
```python
def test_core_cli_help()
def test_core_cli_analyze()
def test_core_cli_validate()
def test_core_cli_version()
def main()
```

#### Class Definitions
- No class definitions
- Core CLI testing
- Comprehensive validation

#### Key Analysis Points
- **Function Count**: 5 functions + 1 main
- **Architecture**: Core CLI testing
- **Scope**: CLI core functionality
- **Dependencies**: Click

---

## Testing Scripts - Processing Engines

### test_dask_engine.py

#### Function Definitions
```python
def create_sample_data(n_samples=1000):
def test_dask_engine_basic():
def test_dask_engine_performance():
def test_dask_engine_memory():
def main()
```

#### Class Definitions
- No class definitions
- Dask engine testing
- Performance validation

#### Key Analysis Points
- **Function Count**: 5 functions + 1 main
- **Architecture**: Dask engine testing framework
- **Scope**: Dask processing engine validation
- **Dependencies**: Dask, Pandas, NumPy

### test_dask_engine_simple.py

#### Function Definitions
```python
def create_test_csv():
def test_dask_engine_simple():
def test_dask_different_schedulers():
def main()
```

#### Class Definitions
- No class definitions
- Simplified Dask testing
- Basic functionality validation

#### Key Analysis Points
- **Function Count**: 4 functions + 1 main
- **Architecture**: Simple Dask testing
- **Scope**: Basic Dask functionality
- **Dependencies**: Dask, Pandas, NumPy

### test_daft_engine.py

#### Function Definitions
```python
def create_sample_data():
def test_daft_engine_basic():
def test_daft_engine_performance():
def test_daft_engine_features():
def main()
```

#### Class Definitions
- No class definitions
- Daft engine testing
- Feature validation

#### Key Analysis Points
- **Function Count**: 5 functions + 1 main
- **Architecture**: Daft engine testing framework
- **Scope**: Daft processing engine validation
- **Dependencies**: Daft, Pandas, NumPy

---

## Testing Scripts - Specialized

### test_lookahead.py

#### Function Definitions
```python
def generate_sample_data(n_samples=1000, n_features=10):
def test_lookahead_bias_prevention():
def test_performance_impact():
def main()
```

#### Class Definitions
- No class definitions
- Lookahead bias testing
- Bias prevention validation

#### Key Analysis Points
- **Function Count**: 4 functions + 1 main
- **Architecture**: Bias prevention testing
- **Scope**: Lookahead bias validation
- **Dependencies**: NumPy, Pandas

### test_inference_engine.py

#### Function Definitions
```python
def test_state_inference():
def test_real_time_prediction():
def test_batch_inference():
def main()
```

#### Class Definitions
- No class definitions
- Inference engine testing
- Real-time prediction validation

#### Key Analysis Points
- **Function Count**: 4 functions + 1 main
- **Architecture**: Inference engine testing
- **Scope**: State prediction validation
- **Dependencies**: NumPy, Pandas

### test_model_persistence.py

#### Function Definitions
```python
def test_model_save_load():
def test_model_metadata():
def test_model_compatibility():
def main()
```

#### Class Definitions
- No class definitions
- Model persistence testing
- Serialization validation

#### Key Analysis Points
- **Function Count**: 4 functions + 1 main
- **Architecture**: Persistence testing framework
- **Scope**: Model save/load validation
- **Dependencies**: Pickle, NumPy, Pandas

---

## Function Complexity Analysis

### High Complexity Functions (>50 lines)
1. **main.py**: `main()` - Complex argument parsing and workflow orchestration
2. **cli.py**: `analyze()` - Rich CLI command with multiple subcommands
3. **cli_comprehensive.py**: `analyze()` - Advanced CLI with extensive options
4. **LSTM.py**: `train_model()` - Neural network training loop with optimization
5. **test_hmm_models.py**: `test_hmm_models()` - Comprehensive model testing
6. **test_backtesting.py**: `main()` - Complete backtesting framework

### Medium Complexity Functions (20-50 lines)
1. **main.py**: `stream_features()` - Streaming data processing with progress tracking
2. **main.py**: `add_features()` - Feature engineering with multiple indicators
3. **hmm_futures_script.py**: Multiple pipeline functions with complete workflows
4. **CLI files**: Command implementations with rich options
5. **Testing files**: Test functions with comprehensive validation

### Low Complexity Functions (<20 lines)
1. **Utility functions**: Helper functions with single responsibilities
2. **Test helpers**: Simple validation and assertion functions
3. **CLI basics**: Basic command implementations
4. **Visualization**: Simple plotting and chart generation

---

## Integration Points Analysis

### Data Processing Integration
- **main.py** → **test_*.py**: Core functionality tested by multiple test files
- **Feature engineering**: Shared across multiple analysis scripts
- **Data validation**: Consistent patterns across implementations

### CLI Integration
- **CLI files** → **main.py**: All CLI files delegate to main.py or similar functionality
- **Configuration**: CLI arguments map to main.py parameters
- **Error handling**: Consistent error handling patterns

### Testing Integration
- **Test files** → **Main files**: Test files validate main directory functionality
- **Test dependencies**: Test files depend on main directory modules
- **Mock objects**: Some tests use mock objects for isolation

### Algorithm Integration
- **LSTM.py**: Potential integration with HMM models
- **Daft files**: Integration with processing engines
- **Script files**: Complete pipelines combining multiple algorithms

---

## Code Organization Patterns

### Main Directory Patterns
1. **Functional Programming**: Global functions, no class structures
2. **Direct Dependencies**: Direct imports from external libraries
3. **CLI Frameworks**: Click-based CLI implementations
4. **Testing Frameworks**: Simple assertion-based testing

### src Directory Patterns (Target)
1. **Object-Oriented**: Class-based design with inheritance
2. **Factory Patterns**: Dynamic model and engine creation
3. **Dependency Injection**: Configurable dependencies
4. **Comprehensive Testing**: Unit, integration, and performance testing

---

*Function Analysis Completed: October 23, 2025*
*Next Step: Task 1.1.1.3 - Analyze CLI implementations*