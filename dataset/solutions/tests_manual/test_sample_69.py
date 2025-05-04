import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_69 import find_common_type


array1 = np.array([1, 2, 3])
array2 = np.array([4.0, 5.0, 6.0])

assert find_common_type(array1, array2) == np.common_type(array1, array2)
