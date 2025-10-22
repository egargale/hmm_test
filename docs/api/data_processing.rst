Data Processing Module
=======================

The :mod:`data_processing` module handles loading, parsing, validating, and feature engineering for OHLCV futures data.

CSV Parser
----------

.. autofunction:: data_processing.csv_parser.process_csv
.. autofunction:: data_processing.csv_parser.standardize_columns
.. autofunction:: data_processing.csv_parser.detect_datetime_column

Feature Engineering
-------------------

.. autofunction:: data_processing.feature_engineering.add_features
.. autofunction:: data_processing.feature_engineering.calculate_returns
.. autofunction:: data_processing.feature_engineering.calculate_moving_averages
.. autofunction:: data_processing.feature_engineering.calculate_volatility

Data Validation
---------------

.. autofunction:: data_processing.data_validation.validate_data
.. autofunction:: data_processing.data_validation.check_data_quality
.. autofunction:: data_processing.data_validation.generate_validation_report

Main Module
-----------

.. automodule:: data_processing
   :members:
   :undoc-members:
   :show-inheritance:
