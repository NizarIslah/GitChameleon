import os

# Add the dataset/samples directory to the Python path
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "samples")
    ),
)

# Import the function to test
from sample_304 import compute_localmin


axis = 0
x = np.array([[1, 0, 1], [2, -1, 0], [2, 1, 3]])

sol = compute_localmin(x, axis)

gt = np.array([[False, False, False], [False, True, True], [False, False, False]])

assert np.array_equal(gt, sol)
