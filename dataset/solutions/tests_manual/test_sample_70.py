import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_70 import find_common_type
import sample_70


array1 = np.array([1, 2, 3])
array2 = np.array([4.0, 5.0, 6.0])

assert find_common_type(array1, array2) == np.find_common_type(array1, array2)
