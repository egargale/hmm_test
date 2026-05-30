# GEMINI.md

## Project Overview
**HMM Futures Analysis** is a sophisticated Python toolkit designed for applying Hidden Markov Models (HMM) to futures market analysis and regime detection. It provides a modular framework for data processing, feature engineering, model training, backtesting, and visualization.

### Key Technologies
- **Language:** Python 3.9+
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Build System:** [hatchling](https://hatch.pypa.io/latest/)
- **Core Libraries:** `hmmlearn`, `scikit-learn`, `pandas`, `numpy`, `scipy`
- **Distributed Processing:** `dask`, `daft`
- **Visualization:** `plotly`, `dash`, `matplotlib`, `seaborn`, `mplfinance`
- **Code Quality:** `ruff`, `mypy`, `pytest`

### Architecture
The project follows a modular structure located in the `src/` directory:
- `data_processing/`: CSV parsing and advanced feature engineering (technical indicators).
- `processing_engines/`: Engines for different data scales (Streaming, Dask, Daft).
- `model_training/`: HMM training, persistence, and state inference.
- `backtesting/`: Framework for testing strategies across market regimes.
- `visualization/`: Interactive charts, dashboards, and regime plotting.
- `pipelines/`: Unified pipelines for end-to-end analysis.
- `utils/`: Configuration management and logging.

---

## Building and Running

### Installation
```bash
# Install core dependencies
uv sync

# Install development dependencies
uv sync --dev

# Install all extras (viz, docs, test)
uv sync --all-extras
```

### Running the Application
The project provides a CLI interface:
```bash
# Analyze futures data
uv run hmm-analyze analyze -i data.csv -o results/

# Validate data format
uv run hmm-analyze validate -i data.csv
```

### Common Commands (via Makefile)
- `make install-dev`: Set up the development environment.
- `make test`: Run the full test suite.
- `make test-cov`: Run tests with coverage report (target: 95%).
- `make format`: Format code using `ruff`.
- `make lint`: Check for linting issues.
- `make type-check`: Run static type checking with `mypy`.
- `make docs`: Build Sphinx documentation.
- `make clean`: Remove build artifacts and cache files.

---

## Development Conventions

### Code Style & Quality
- **Formatting:** Strictly enforced via `ruff format`.
- **Linting:** Enforced via `ruff check`.
- **Type Hints:** Required for all new code; verified with `mypy`.
- **Import Sorting:** Handled automatically by `ruff`.

### Testing Practices
- **Framework:** `pytest` is used for all testing.
- **Coverage:** The project aims for 95%+ code coverage.
- **Test Data:** For tests requiring data, prefer sampling from local CSV files (e.g., `test_data/`) to save memory.
- **Markers:** Use markers like `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow` to categorize tests.

### Workflow
- **Dependency Management:** Always use `uv` and `uvx` for managing libraries and running tools.
- **Task Automation:** Use the `Makefile` for common development tasks.
- **Pre-commit:** Install pre-commit hooks using `uv run pre-commit install`.
- **Documentation:** Maintain Sphinx documentation in the `docs/` directory.

### Specialized Instructions (from CLAUDE.md)
- Always search for local CSV files for test data samples.
- Use `uv` and `uvx` as the primary library managers.
- Follow the Task Master AI development workflow as specified in `.taskmaster/CLAUDE.md`.
