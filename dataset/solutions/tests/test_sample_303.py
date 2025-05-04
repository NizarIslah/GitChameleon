import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_303 import compute_localmin


class TestSample303(unittest.TestCase):
    def test_compute_localmin_1d(self):
        """Test compute_localmin function with 1D array."""
        # Create a test array with known minima
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
        # Expected minima are at indices 1, 3, 6, and 8
        expected = np.array([False, True, False, True, False, False, True, False, True])

        result = compute_localmin(x, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_compute_localmin_2d(self):
        """Test compute_localmin function with 2D array along axis 0."""
        # Create a 2D test array
        x = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])

        # Expected minima along axis 0
        expected_axis0 = np.array(
            [[False, False, False], [True, False, False], [False, False, True]]
        )

        result_axis0 = compute_localmin(x, axis=0)

        np.testing.assert_array_equal(result_axis0, expected_axis0)

    def test_compute_localmin_2d_axis1(self):
        """Test compute_localmin function with 2D array along axis 1."""
        # Create a 2D test array
        x = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])

        # Expected minima along axis 1
        expected_axis1 = np.array(
            [[False, True, False], [False, False, False], [False, False, True]]
        )

        result_axis1 = compute_localmin(x, axis=1)

        np.testing.assert_array_equal(result_axis1, expected_axis1)

    def test_compute_localmin_empty(self):
        """Test compute_localmin function with empty array."""
        x = np.array([])
        result = compute_localmin(x, axis=0)
        self.assertEqual(result.size, 0)

    def test_compute_localmin_flat(self):
        """Test compute_localmin function with flat array (no minima)."""
        x = np.array([5, 5, 5, 5, 5])
        expected = np.array([False, False, False, False, False])

        result = compute_localmin(x, axis=0)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
