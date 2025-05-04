import os

# Add the directory containing sample_126.py to the Python path
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "solutions"))

# Import the function to test
from sample_126 import compute_lanczos_window


window_size = 31
window = compute_lanczos_window(window_size)
window_numpy = 2 * np.arange(window_size) / (window_size - 1) - 1
window_numpy = np.sinc(window_numpy)
window_numpy = window_numpy / np.max(window_numpy)
assertion_value = np.allclose(window, window_numpy)
assert assertion_value
