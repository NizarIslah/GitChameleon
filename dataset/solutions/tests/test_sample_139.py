import os
import sys
import unittest

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_139 import apply_maximum_filter


class TestMaximumFilter(unittest.TestCase):
    def test_apply_maximum_filter_basic(self):
        """Test basic functionality of apply_maximum_filter."""
        input_array = np.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
        )
        result = apply_maximum_filter(input_array, size=3)
        # Use scipy's maximum_filter to generate the expected result
        expected = maximum_filter(input_array, size=3, axes=[1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_apply_maximum_filter_different_size(self):
        """Test maximum filter with a different filter size."""
        input_array = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]],
            ]
        )
        result = apply_maximum_filter(input_array, size=2)
        expected = maximum_filter(input_array, size=2, axes=[1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_apply_maximum_filter_with_zeros(self):
        """Test maximum filter with an array containing zeros."""
        input_array = np.zeros((2, 3, 3))
        input_array[0, 1, 1] = 5  # Set one value to non-zero
        result = apply_maximum_filter(input_array, size=2)
        expected = maximum_filter(input_array, size=2, axes=[1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_apply_maximum_filter_compare_with_direct(self):
        """Compare our function with direct call to scipy's maximum_filter."""
        np.random.seed(42)  # For reproducibility
        input_array = np.random.rand(3, 5, 5)
        result = apply_maximum_filter(input_array, size=3)
        expected = maximum_filter(input_array, size=3, axes=[1, 2])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
