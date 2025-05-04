import unittest
import numpy as np
from sklearn.impute import SimpleImputer
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_50 import get_imputer

data = np.array([[1, 2, 3], [4, None, 6], [7, 8, None]], dtype=float)
expected_type = SimpleImputer
assert isinstance(get_imputer(data), expected_type)
