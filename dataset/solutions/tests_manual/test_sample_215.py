import unittest
import numpy as np
from scipy.stats import iqr
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_215 import custom_iqr

data_array = np.array([1, 2, 3, 4, 5])

computed_iqr = custom_iqr(data_array)
expect = 2
assert computed_iqr == expect
