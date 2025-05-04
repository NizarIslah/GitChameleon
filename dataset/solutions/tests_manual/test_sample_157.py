import os

# Add the parent directory to the path so we can import the module
import sys
import unittest
import warnings

import numpy as np
from scipy.linalg import det

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_157 import check_invertibility


matrices = np.array([[[1, 2], [3, 4]], [[0, 1], [1, 0]], [[2, 0], [0, 2]]])
assertion_value = check_invertibility(matrices)
assert assertion_value
matrices = np.array(
    [[[1, 2], [3, 4]], [[0, 1], [1, 0]], [[2, 0], [0, 2]], [[0, 0], [0, 0]]]
)
assertion_value = not check_invertibility(matrices)
assert assertion_value
