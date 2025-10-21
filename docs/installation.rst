Installation
============

This guide will help you install and set up the HMM Futures Analysis system.

System Requirements
------------------

* Python 3.8 or higher
* 8GB+ RAM recommended for large datasets
* 2GB+ disk space

Installation Methods
--------------------

Method 1: Using UV (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UV is the recommended package manager for this project.

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/egargale/hmm_test.git
   cd hmm_test

   # Install dependencies and project
   uv install

Method 2: Using Pip
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/egargale/hmm_test.git
   cd hmm_test

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -e .

Dependencies
------------

The project automatically installs all required dependencies including:

**Core Libraries:**
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning utilities
- `hmmlearn`: Hidden Markov Model implementation

**Processing Engines:**
- `dask[distributed]`: Distributed computing
- `daft`: Out-of-core DataFrame processing
- `click`: Command-line interface

**Visualization:**
- `matplotlib`: Plotting and charts
- `mplfinance`: Financial charting
- `plotly`: Interactive dashboards
- `jinja2`: Template engine

**Development Tools:**
- `pytest`: Testing framework
- `sphinx`: Documentation generation
- `black`: Code formatting
- `mypy`: Type checking

Optional Dependencies
--------------------

For development and additional features:

.. code-block:: bash

   # Development dependencies
   pip install pytest-cov black mypy flake8

   # Additional visualization options
   pip install seaborn plotly-dash

   # Performance monitoring
   pip install psutil memory-profiler

Verification
------------

To verify your installation:

.. code-block:: python

   # Test basic imports
   import numpy as np
   import pandas as pd
   from src.utils import get_logger

   # Test CLI
   python cli_simple.py --version

   # Run basic tests
   python -m pytest tests/ -v

Troubleshooting
---------------

Common Issues and Solutions:

**Import Errors:**
If you encounter import errors, ensure the `src` directory is in your Python path:

.. code-block:: python

   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path('.').absolute() / 'src'))

**Memory Issues:**
For large datasets, consider using the Dask engine:

.. code-block:: python

   python cli_simple.py analyze -i large_data.csv --engine dask

**Visualization Errors:**
If matplotlib fails to display plots, try:

.. code-block:: bash

   export MPLBACKEND=Agg  # For headless environments
   # or
   pip install PyQt5  # For GUI environments

**Installation on Windows:**
Some packages may require Microsoft Visual C++ Build Tools. Install them from the Microsoft website or use conda:

.. code-block:: bash

   conda install numpy pandas scikit-learn hmmlearn

Docker Installation
------------------

For containerized environments:

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       && rm -rf /var/lib/apt/lists/*

   # Copy project files
   COPY . /app

   # Install with uv
   RUN pip install uv
   RUN uv install

   # Set entrypoint
   ENTRYPOINT ["python", "cli_simple.py"]

   # Build image
   # docker build -t hmm-analysis .

   # Run container
   # docker run -v $(pwd)/data:/app/data hmm-analysis analyze -i /app/data/sample.csv