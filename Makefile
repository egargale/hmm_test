.PHONY: help install install-dev test lint format type-check security clean docs docs-serve build publish

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  type-check   Run type checking"
	@echo "  security     Run security checks"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  build        Build package"
	@echo "  publish      Publish package to PyPI"

# Installation
install:
	uv sync

install-dev:
	uv sync --dev

install-all:
	uv sync --all-extras

# Testing
test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing -v

test-fast:
	uv run pytest -v -m "not slow"

test-integration:
	uv run pytest -v -m integration

# Code quality
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/ --ignore-missing-imports

security:
	uv run bandit -r src/ -f json -o bandit-report.json
	uv run safety check

# Combined quality checks
check: lint type-check security test

quality: format check

# Documentation
docs:
	cd docs && uv run sphinx-build -b html . _build/html

docs-serve:
	cd docs && uv run sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

docs-linkcheck:
	cd docs && uv run sphinx-build -b linkcheck . _build/linkcheck

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

# Local installation for testing
install-local: build
	pip install dist/*.whl --force-reinstall

# Publishing (requires proper authentication)
publish-test: build
	uv run twine upload --repository testpypi dist/*

publish: build
	uv run twine upload dist/*

# Development workflow
dev-setup: install-dev
	uv run pre-commit install

# Quick development cycle
dev: format test

# Full CI simulation
ci: lint type-check security test-cov docs

# Performance profiling
profile:
	uv run python -m cProfile -o profile.stats src/cli_simple.py analyze -i test_data/test_futures.csv -o output/

# Memory profiling
memory-profile:
	uv run python -m memory_profiler src/cli_simple.py analyze -i test_data/test_futures.csv -o output/

# Benchmarking
benchmark:
	uv run pytest tests/benchmarks/ -v

# Docker (if applicable)
docker-build:
	docker build -t hmm-futures-analysis .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data hmm-futures-analyze

# Release workflow
version-patch:
	@echo "Bumping patch version..."
	@uv run hatch version patch

version-minor:
	@echo "Bumping minor version..."
	@uv run hatch version minor

version-major:
	@echo "Bumping major version..."
	@uv run hatch version major

release: clean check docs build
	@echo "Ready to release! Run 'make publish' to upload to PyPI."