import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_140 import apply_maximum_filter


class TestApplyMaximumFilter(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the function works on a simple 2D array."""
        # Create a simple 2D array
        input_array = np.array([[1, 2, 3, 2, 1], [5, 6, 7, 6, 5]])
        size = 3

        # Expected output: maximum_filter with size=3 applied to each row
        expected_output = np.array([[2, 3, 3, 3, 2], [6, 7, 7, 7, 6]])

        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)

    def test_single_row(self):
        """Test with a single row array."""
        input_array = np.array([[1, 2, 3, 4, 5]])
        size = 3

        # Expected output: maximum_filter with size=3 applied to the row
        expected_output = np.array([[2, 3, 4, 5, 5]])

        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)

    def test_with_zeros(self):
        """Test with an array containing zeros."""
        input_array = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])
        size = 3

        expected_output = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])

        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)

    def test_empty_array(self):
        """Test with an empty array."""
        input_array = np.array([])
        size = 3

        result = apply_maximum_filter(input_array, size)
        self.assertEqual(result.size, 0)


if __name__ == "__main__":
    unittest.main()
