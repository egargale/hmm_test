"""
Data Types Module

Defines type aliases for the HMM futures analysis project.
"""

import numpy as np
import pandas as pd

# Type aliases for better readability
PriceData = pd.DataFrame  # Expected columns: open, high, low, close, volume
FeatureMatrix = np.ndarray
StateSequence = np.ndarray
ProbabilityMatrix = np.ndarray
