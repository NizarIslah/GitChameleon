import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_53 import get_pairwise_dist

X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[1, 1], [4, 4]])
expected_result = np.array([1, 5, 5, 1, 9, 3])
assert np.allclose(get_pairwise_dist(X, Y), expected_result, atol=1e-3)
