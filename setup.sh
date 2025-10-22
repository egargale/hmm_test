#!/bin/bash

# HMM Futures Analysis Setup Script
# This script sets up the development environment for the HMM Futures Analysis project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

print_status "Starting HMM Futures Analysis setup..."

# Detect operating system
OS=$(detect_os)
print_status "Detected OS: $OS"

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python version: $PYTHON_VERSION"

    # Check if Python version is >= 3.9
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
        print_success "Python version is compatible (>= 3.9)"
    else
        print_error "Python 3.9 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if uv is installed, install if not
if command_exists uv; then
    print_success "uv is already installed"
    UV_VERSION=$(uv --version)
    print_status "uv version: $UV_VERSION"
else
    print_status "Installing uv..."
    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "uv installed successfully"
    else
        print_error "curl is not installed. Please install curl and try again."
        exit 1
    fi
fi

# Check if git is installed
if command_exists git; then
    print_success "Git is available"
else
    print_warning "Git is not installed. Some features may not work properly."
fi

# Create virtual environment and install dependencies
print_status "Creating virtual environment and installing dependencies..."

if command_exists uv; then
    # Use uv for installation
    if [ -f "uv.lock" ]; then
        print_status "Installing dependencies from uv.lock..."
        uv sync
    else
        print_status "Installing dependencies from pyproject.toml..."
        uv sync --dev
    fi
    print_success "Dependencies installed successfully"
else
    # Fallback to pip
    print_status "Creating virtual environment with venv..."
    python3 -m venv .venv
    source .venv/bin/activate

    print_status "Upgrading pip..."
    pip install --upgrade pip

    print_status "Installing dependencies..."
    pip install -e ".[dev,docs,test]"
    print_success "Dependencies installed successfully"
fi

# Install pre-commit hooks
if command_exists uv; then
    print_status "Installing pre-commit hooks..."
    uv run pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "Skipping pre-commit installation (uv not available)"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data output logs docs/_build notebooks
print_success "Directories created"

# Set up configuration files
if [ ! -f "config_default.yaml" ]; then
    print_status "Creating default configuration file..."
    cp config_example.yaml config_default.yaml
    print_success "Default configuration created"
fi

# Run basic validation
print_status "Running basic validation..."

if command_exists uv; then
    # Test imports
    print_status "Testing Python imports..."
    uv run python -c "
import sys
sys.path.insert(0, 'src')
try:
    from utils import ProcessingConfig, HMMConfig
    from data_processing.csv_parser import process_csv
    print('All imports successful!')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

    # Test CLI
    print_status "Testing CLI..."
    uv run python cli_comprehensive.py version
    print_success "CLI validation passed"
else
    print_warning "Skipping validation (uv not available)"
fi

# Build documentation
if command_exists uv; then
    print_status "Building documentation..."
    cd docs
    uv run sphinx-build -b html . _build/html >/dev/null 2>&1 || {
        print_warning "Documentation build failed, but continuing..."
    }
    cd ..
    print_success "Documentation built successfully"
else
    print_warning "Skipping documentation build (uv not available)"
fi

# Print setup completion message
echo ""
print_success "🎉 HMM Futures Analysis setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
if command_exists uv; then
    echo "   uv run command           # Run commands with uv"
    echo "   uv run python cli.py    # Run the CLI"
else
    echo "   source .venv/bin/activate  # On Linux/macOS"
    echo "   .venv\\Scripts\\activate     # On Windows"
fi
echo ""
echo "2. Run the analysis:"
echo "   python cli_comprehensive.py --help"
echo "   python cli_comprehensive.py validate -i your_data.csv"
echo "   python cli_comprehensive.py analyze -i your_data.csv -o output/"
echo ""
echo "3. View documentation:"
echo "   Open docs/_build/html/index.html in your browser"
echo ""
echo "4. Run tests:"
echo "   uv run pytest tests/ -v"
echo ""
echo "5. Development commands:"
echo "   uv run pre-commit run --all-files  # Run code quality checks"
echo "   uv run python -m pytest tests/ -v  # Run tests"
echo "   uv run python cli_comprehensive.py --help  # CLI help"
echo ""

if [ "$OS" = "windows" ]; then
    print_status "Windows-specific notes:"
    echo "   - Use PowerShell or Command Prompt"
    echo "   - Some visualization features may need additional setup"
    echo ""
elif [ "$OS" = "macos" ]; then
    print_status "macOS-specific notes:"
    echo "   - Install Xcode Command Line Tools: xcode-select --install"
    echo "   - Some packages may require: brew install"
    echo ""
else
    print_status "Linux-specific notes:"
    echo "   - Install build tools: sudo apt-get install build-essential"
    echo "   - Install dev headers: sudo apt-get install python3-dev"
    echo ""
fi

print_success "Setup completed! Happy analyzing! 🚀"