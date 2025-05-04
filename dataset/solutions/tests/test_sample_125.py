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

        expected = windows.lanczos(window_size)

        # Check that all values match
        np.testing.assert_allclose(result, expected)

    def test_different_sizes(self):
        """Test that the function works with different window sizes."""
        sizes = [1, 5, 10, 20]
        for size in sizes:
            result = compute_lanczos_window(size)
            self.assertEqual(len(result), size)

    def test_window_properties(self):
        """Test some known properties of the Lanczos window."""
        # For odd window sizes, the center value should be 1.0
        odd_size = 11
        result_odd = compute_lanczos_window(odd_size)
        self.assertAlmostEqual(result_odd[odd_size // 2], 1.0)

        # The window should be symmetric
        window_size = 10
        result = compute_lanczos_window(window_size)
        for i in range(window_size // 2):
            self.assertAlmostEqual(result[i], result[window_size - 1 - i])


if __name__ == "__main__":
    unittest.main()
