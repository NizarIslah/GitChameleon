import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.ndimage import percentile_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_132 import apply_percentile_filter


class TestPercentileFilter(unittest.TestCase):
    def test_apply_percentile_filter_basic(self):
        """Test basic functionality with a simple 2D array."""
        # Create a simple 2D array
        test_array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        # Apply the filter with 50th percentile (median) and size 3
        result = apply_percentile_filter(test_array, percentile=50, size=3)

        # Calculate expected result manually
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = percentile_filter(test_array[i], percentile=50, size=3)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def test_apply_percentile_filter_different_percentiles(self):
        """Test with different percentile values."""
        # Create a test array
        test_array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

        # Test with different percentiles
        for percentile in [0, 25, 50, 75, 100]:
            result = apply_percentile_filter(test_array, percentile=percentile, size=3)

            # Calculate expected result
            expected = np.zeros_like(test_array)
            for i in range(test_array.shape[0]):
                expected[i] = percentile_filter(
                    test_array[i], percentile=percentile, size=3
                )

            # Check if the result matches the expected output
            np.testing.assert_array_equal(
                result, expected, err_msg=f"Failed with percentile={percentile}"
            )

    def test_apply_percentile_filter_different_sizes(self):
        """Test with different filter sizes."""
        # Create a test array
        test_array = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])

        # Test with different filter sizes
        for size in [1, 3, 5]:
            result = apply_percentile_filter(test_array, percentile=50, size=size)

            # Calculate expected result
            expected = np.zeros_like(test_array)
            for i in range(test_array.shape[0]):
                expected[i] = percentile_filter(test_array[i], percentile=50, size=size)

            # Check if the result matches the expected output
            np.testing.assert_array_equal(
                result, expected, err_msg=f"Failed with size={size}"
            )

    def test_apply_percentile_filter_random_array(self):
        """Test with a random array to ensure robustness."""
        # Set a seed for reproducibility
        np.random.seed(42)

        # Create a random array
        test_array = np.random.rand(5, 10) * 100

        # Apply the filter
        result = apply_percentile_filter(test_array, percentile=75, size=3)

        # Calculate expected result
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = percentile_filter(test_array[i], percentile=75, size=3)

        # Check if the result matches the expected output
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_percentile_filter_edge_cases(self):
        """Test edge cases like single row arrays and float percentiles."""
        # Test with a single row array
        test_array = np.array([[1, 2, 3, 4, 5]])

        result = apply_percentile_filter(test_array, percentile=50, size=3)

        expected = np.zeros_like(test_array)
        expected[0] = percentile_filter(test_array[0], percentile=50, size=3)

        np.testing.assert_array_equal(result, expected)

        # Test with float percentile
        test_array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        result = apply_percentile_filter(test_array, percentile=33.3, size=3)

        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = percentile_filter(test_array[i], percentile=33.3, size=3)

        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
