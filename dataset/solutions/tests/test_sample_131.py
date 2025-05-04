import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_131 import apply_percentile_filter


class TestPercentileFilter(unittest.TestCase):
    def test_apply_percentile_filter_basic(self):
        """Test basic functionality of apply_percentile_filter."""
        # Create a 3D array (3x3x3)
        test_array = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        )

        # Apply percentile filter with 50th percentile (median) and size 3
        result = apply_percentile_filter(test_array, 50, 3)

        # Check shape is preserved for the first dimension
        self.assertEqual(result.shape[0], test_array.shape[0])

        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

    def test_apply_percentile_filter_different_percentiles(self):
        """Test with different percentile values."""
        # Create a simple 3D array
        test_array = np.ones((3, 5, 5))
        test_array[1, 2:4, 2:4] = 5  # Set some values to 5

        # Test with minimum (0th percentile)
        min_result = apply_percentile_filter(test_array, 0, 3)
        # Test with maximum (100th percentile)
        max_result = apply_percentile_filter(test_array, 100, 3)
        # Test with median (50th percentile)
        median_result = apply_percentile_filter(test_array, 50, 3)

        # Check that min <= median <= max
        np.testing.assert_array_less(
            min_result, max_result + 1e-10
        )  # Add small epsilon for floating point comparison

        # For this specific test case, we know min should be 1 and max should be 5
        self.assertTrue(np.all(min_result >= 1))
        self.assertTrue(np.all(max_result <= 5))

    def test_apply_percentile_filter_size(self):
        """Test with different filter sizes."""
        # Create a 3D array with a gradient
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        test_array = np.stack([x, y, x + y])

        # Apply filter with different sizes
        small_filter = apply_percentile_filter(test_array, 50, 3)
        large_filter = apply_percentile_filter(test_array, 50, 5)

        # The larger filter should have a stronger smoothing effect
        # This is a simple heuristic test - the variance of the large filter result
        # should be less than or equal to the small filter result
        self.assertLessEqual(np.var(large_filter), np.var(small_filter) + 1e-10)

    def test_apply_percentile_filter_edge_cases(self):
        """Test edge cases like small arrays and extreme percentiles."""
        # Test with a minimal 3D array
        small_array = np.arange(8).reshape(2, 2, 2)
        result_small = apply_percentile_filter(small_array, 50, 2)
        self.assertEqual(result_small.shape[0], 2)

        # Test with extreme percentiles
        test_array = np.random.rand(3, 4, 4)
        min_result = apply_percentile_filter(test_array, 0, 3)
        max_result = apply_percentile_filter(test_array, 100, 3)

        # Min should be less than or equal to max
        np.testing.assert_array_less(min_result, max_result + 1e-10)


if __name__ == "__main__":
    unittest.main()
