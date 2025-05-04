import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_68 import apply_correlate_full


arr1 = np.array([1, 2, 3])
arr2 = np.array([0, 1, 0.5])
assert apply_correlate_full(arr1, arr2).all() == np.correlate(arr1, arr2, "full").all()
