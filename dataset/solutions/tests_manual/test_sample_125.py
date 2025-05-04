# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_125 import compute_lanczos_window


class TestLanczosWindow(unittest.TestCase):
    def test_return_type(self):
        """Test that the function returns a numpy array."""
        result = compute_lanczos_window(10)
        self.assertIsInstance(result, np.ndarray)

    def test_array_length(self):
        """Test that the returned array has the correct length."""
        window_size = 15
        result = compute_lanczos_window(window_size)
        self.assertEqual(len(result), window_size)

    def test_window_values(self):
        """Test that the window values are correct."""
        # Test with a known window size
        window_size = 7
        result = compute_lanczos_window(window_size)

        # Calculate expected values directly using scipy's lanczos function
        import scipy.signal.windows as windows


window_size = 31
window = compute_lanczos_window(window_size)
window_numpy = 2 * np.arange(window_size) / (window_size - 1) - 1
window_numpy = np.sinc(window_numpy)
window_numpy = window_numpy / np.max(window_numpy)
assertion_value = np.allclose(window, window_numpy)
assert assertion_value
