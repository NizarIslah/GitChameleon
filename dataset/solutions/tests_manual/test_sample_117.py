import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_117 import compute_circular_variance


a = np.array([0, 2 * np.pi / 3, 5 * np.pi / 3])
output = compute_circular_variance(a)
assertion_value = np.allclose(output, 1 - np.abs(np.mean(np.exp(1j * a))))
assert assertion_value
