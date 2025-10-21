HMM Futures Analysis
===================

Welcome to the HMM Futures Analysis documentation!

This comprehensive system provides advanced Hidden Markov Model (HMM) analysis for futures market regime detection, backtesting, and performance optimization.

.. image:: https://img.shields.io/badge/Version-1.0.0-blue.svg
   :target: https://github.com/egargale/hmm_test
   :alt: Version

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Features
--------

* **Multi-Engine Processing**: Support for streaming, Dask, and Daft processing engines
* **Advanced HMM Training**: Robust Hidden Markov Model training with multiple restarts
* **Regime Detection**: Automatic identification of market regimes and state transitions
* **Backtesting Engine**: Comprehensive backtesting with realistic transaction costs
* **Performance Analytics**: Advanced risk-adjusted performance metrics and bias prevention
* **Visualization**: Professional charts, interactive dashboards, and detailed reports
* **CLI Interface**: Command-line tool for complete analysis workflows

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/egargale/hmm_test.git
   cd hmm_test

   # Install dependencies with uv
   uv install

   # Or with pip
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from src.data_processing.csv_parser import process_csv
   from src.data_processing.feature_engineering import add_features
   from src.model_training.hmm_trainer import train_model
   from src.model_training.inference_engine import StateInference

   # Load and process data
   data = process_csv('your_data.csv')
   features = add_features(data)

   # Train HMM model
   config = {'n_components': 3, 'n_iter': 100, 'random_state': 42}
   model, metadata = train_model(features, config)

   # Infer states
   inference = StateInference(model)
   states = inference.infer_states(features['close'].values.reshape(-1, 1))

CLI Usage
~~~~~~~~~~

.. code-block:: bash

   # Validate data
   python cli_simple.py validate -i data.csv -o output/

   # Run complete analysis
   python cli_simple.py analyze -i data.csv -o output/ -n 3

   # With custom options
   python cli_simple.py analyze \
       -i data.csv \
       -o results/ \
       --n-states 4 \
       --test-size 0.2 \
       --random-seed 42

Table of Contents
================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide/index
   api/index
   examples/index
   developer_guide/index
   contributing

   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`