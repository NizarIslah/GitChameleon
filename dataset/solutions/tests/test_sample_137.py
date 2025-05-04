import os
import sys
import unittest

import numpy as np
from scipy.ndimage import minimum_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_137 import apply_minimum_filter


class TestMinimumFilter(unittest.TestCase):
    def test_apply_minimum_filter_basic(self):
        """Test basic functionality of apply_minimum_filter."""
        # Create a 3D array (3x3x3)
        test_array = np.array(
            [[[5, 6, 7], [8, 9, 10], [11, 12, 13]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
        )

        # Apply minimum filter with size 2
        result = apply_minimum_filter(test_array, 2)

        # Expected result: minimum_filter applied to axes 1 and 2 with size 2
        expected = minimum_filter(test_array, size=2, axes=[1, 2])

        # Check if result matches expected
        np.testing.assert_array_equal(result, expected)

    def test_apply_minimum_filter_different_sizes(self):
        """Test minimum filter with different filter sizes."""
        # Create a 3D array (2x4x4)
        test_array = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
            ]
        )

        # Test with size 1 (should return the original array)
        result_size_1 = apply_minimum_filter(test_array, 1)
        expected_size_1 = minimum_filter(test_array, size=1, axes=[1, 2])
        np.testing.assert_array_equal(result_size_1, expected_size_1)

        # Test with size 3
        result_size_3 = apply_minimum_filter(test_array, 3)
        expected_size_3 = minimum_filter(test_array, size=3, axes=[1, 2])
        np.testing.assert_array_equal(result_size_3, expected_size_3)

    def test_apply_minimum_filter_edge_cases(self):
        """Test minimum filter with edge cases."""
        # Test with a 3D array with one dimension of size 1
        test_array_thin = np.array([[[1, 2, 3]]])
        result_thin = apply_minimum_filter(test_array_thin, 2)
        expected_thin = minimum_filter(test_array_thin, size=2, axes=[1, 2])
        np.testing.assert_array_equal(result_thin, expected_thin)

        # Test with array of all same values
        test_array_same = np.ones((2, 3, 3))
        result_same = apply_minimum_filter(test_array_same, 2)
        expected_same = minimum_filter(test_array_same, size=2, axes=[1, 2])
        np.testing.assert_array_equal(result_same, expected_same)

    def test_apply_minimum_filter_random_data(self):
        """Test minimum filter with random data."""
        # Set a seed for reproducibility
        np.random.seed(42)

        # Create a random 3D array
        test_array = np.random.randint(0, 100, size=(3, 5, 5))

        # Apply minimum filter with size 3
        result = apply_minimum_filter(test_array, 3)

        # Expected result
        expected = minimum_filter(test_array, size=3, axes=[1, 2])

        # Check if result matches expected
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
