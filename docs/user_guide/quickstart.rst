Quickstart Guide
===============

Get up and running with HMM Futures Analysis in just a few minutes. This guide will walk you through your first complete analysis from data loading to visualization.

Prerequisites
-------------

Before you begin, make sure you have:

* Python 3.8+ installed
* The HMM Futures Analysis code installed (see :doc:`../installation`)
* A sample futures dataset in CSV format

Step 1: Install the System
--------------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/egargale/hmm_test.git
   cd hmm_test

   # Install dependencies
   uv install

   # Verify installation
   python cli_simple.py --version

Step 2: Prepare Your Data
--------------------------

Your data should be in CSV format with OHLCV (Open, High, Low, Close, Volume) columns. Here's what a properly formatted file looks like:

.. code-block:: csv

   datetime,open,high,low,close,volume
   2020-01-01,100.50,102.00,99.75,101.25,15000
   2020-01-02,101.25,103.50,100.00,102.75,18000
   2020-01-03,102.75,104.00,101.50,103.00,16500
   ...

If you don't have data, the system includes sample data:

.. code-block:: bash

   # Use the included test data
   python cli_simple.py analyze -i test_data/test_futures.csv -o output/

Step 3: Validate Your Data
--------------------------

First, ensure your data is properly formatted:

.. code-block:: bash

   # Validate data format
   python cli_simple.py validate -i your_data.csv -o validation_output/

This will check for:
- Required OHLCV columns
- Data type consistency
- Missing values
- Outliers and anomalies
- Date range and frequency

If validation passes, you'll see a message like:

.. code-block:: text

   ✅ Data validation passed!
   📊 1000 rows of data
   📅 Date range: 2020-01-01 to 2022-12-31
   📈 Columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']
   📄 Validation report saved to: validation_output/validation_report.txt

Step 4: Run Your First Analysis
------------------------------

Now run the complete HMM analysis pipeline:

.. code-block:: bash

   # Basic analysis with default settings
   python cli_simple.py analyze -i your_data.csv -o results/

This command will:

1. **Load and validate** your data
2. **Engineer features** (returns, moving averages, volatility)
3. **Train an HMM model** with 3 states
4. **Infer hidden states** for the entire timeline
5. **Generate outputs** including states, statistics, and reports

You should see output like:

.. code-block:: text

   🚀 Starting Simple HMM Analysis Pipeline
   📂 Input file: your_data.csv
   📊 Output directory: results/
   🔢 Number of states: 3

   📁 Step 1: Loading and validating data...
   ✅ Loaded 1000 rows of data

   ⚙️  Step 2: Basic feature engineering...
   ✅ Added 6 basic features

   🧠 Step 3: Simple HMM model training...
   Training HMM with 3 states...
   [====================] 100% 50/50 [00:02<00:00,  10.00it/s]
   ✅ HMM training completed. Score: -1250.43

   🔍 Step 4: Simple state inference...
   ✅ State inference completed. Found 3 unique states

   💾 Step 5: Saving results...
   ✅ Results saved to results/
     📊 States: results/states.csv
     🧠 Model info: results/model_info.txt
     📈 Statistics: results/state_statistics.txt

   🎉 SIMPLE HMM ANALYSIS COMPLETED!
   ============================================================
   📂 Results saved to: results/
   📊 Data processed: 750 rows
   🔢 HMM states: 3
   ⏱️  Total time: 12.45 seconds

Step 5: Explore the Results
---------------------------

The analysis generates several output files:

**States File (states.csv):**
Contains the inferred hidden states for each timestamp:

.. code-block:: python

   import pandas as pd
   states = pd.read_csv('results/states.csv')
   print(states.head())

   # Output:
   #   datetime  open   high    low   close  volume  hmm_state
   # 2020-01-01  100.50 102.00  99.75  101.25  15000  1
   # 2020-01-02  101.25 103.50 100.00  102.75  18000  1
   # 2020-01-03  102.75 104.00 101.50 103.00  16500  2
   # ...

**Model Information (model_info.txt):**
Details about the trained HMM model:

.. code-block:: text

   HMM Model Information
   =====================

   n_components: 3
   n_features: 1
   converged: True
   n_iter: 50
   score: -1250.43
   training_samples: 750
   total_samples: 750

**State Statistics (state_statistics.txt):**
Analysis of each regime's characteristics:

.. code-block:: text

   State Statistics
   ================

   State 0:
     count: 425.0000
     mean_return: 0.0012
     std_return: 0.0089

   State 1:
     count: 200.0000
     mean_return: -0.0003
     std_return: 0.0156

   State 2:
     count: 125.0000
     mean_return: 0.0045
     std_return: 0.0234

Step 6: Customize Your Analysis
----------------------------

Experiment with different configurations:

**Different Number of States:**

.. code-block:: bash

   # Try 4 states instead of 3
   python cli_simple.py analyze -i your_data.csv -o results_4states/ --n-states 4

**Test/Train Split:**

.. code-block:: bash

   # Use 30% of data for testing
   python cli_simple.py analyze -i your_data.csv -o results/ --test-size 0.3

**Random Seed for Reproducibility:**

.. code-block:: bash

   # Set a specific random seed
   python cli_simple.py analyze -i your_data.csv -o results/ --random-seed 123

Step 7: Advanced Usage
--------------------

For larger datasets, use the Dask engine:

.. code-block:: bash

   # Process large datasets with Dask
   python cli_simple.py analyze -i large_dataset.csv -o results/ --engine dask

Add visualization and reporting:

.. code-block:: bash

   # Generate all outputs (slower but comprehensive)
   python cli_simple.py analyze -i your_data.csv -o results/ \
       --generate-charts --generate-dashboard --generate-report

This will create:
- **Chart files**: PNG images of state visualizations
- **Dashboard**: Interactive HTML dashboard
- **Report**: Comprehensive analysis report

Step 8: Use the Python API
---------------------------

For programmatic access, use the Python API:

.. code-block:: python

   from src.data_processing.csv_parser import process_csv
   from src.data_processing.feature_engineering import add_features
   from src.model_training.hmm_trainer import train_model
   from src.model_training.inference_engine import StateInference

   # Load data
   data = process_csv('your_data.csv')

   # Add features
   features = add_features(data)

   # Train model
   config = {
       'n_components': 3,
       'n_iter': 100,
       'random_state': 42
   }
   model, metadata = train_model(
       features['close'].values.reshape(-1, 1),
       config=config
   )

   # Infer states
   inference = StateInference(model)
   states = inference.infer_states(features['close'].values.reshape(-1, 1))

   print(f"Trained model with {metadata['n_components']} states")
   print(f"State sequence: {states[:10]}")  # First 10 states

What's Next?
-------------

Congratulations! You've completed your first HMM analysis. Here are suggested next steps:

1. **Explore State Analysis**: Dive deeper into regime characteristics
2. **Customize Features**: Add your own technical indicators
3. **Backtest Strategies**: Implement and test trading strategies
4. **Advanced Visualization**: Create custom charts and dashboards
5. **Batch Processing**: Automate analysis of multiple datasets

For detailed guidance on these topics, continue reading the user guide sections.

Troubleshooting
---------------

**Common Issues:**

* **"No such file or directory"**: Check that your data file path is correct
* **"Memory error"**: Use the Dask engine (`--engine dask`) for large datasets
* **"Training failed"**: Try fewer states (`--n-states 2`) or more data
* **"Import errors"**: Ensure you're running from the project root directory

**Getting Help:**

* Check the :doc:`../user_guide/troubleshooting` guide
* Review the command-line help: `python cli_simple.py --help`
* Examine log files for detailed error messages