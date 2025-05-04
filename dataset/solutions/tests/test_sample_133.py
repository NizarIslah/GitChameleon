import os
import sys
import unittest

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_133 import apply_median_filter


class TestMedianFilter(unittest.TestCase):
    def test_apply_median_filter_3d_array(self):
        """Test median filter on a 3D array with known values."""
        # Create a 3D test array with shape (2, 3, 3)
        test_array = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ]
        )

        # Apply our function with filter size 3
        result = apply_median_filter(test_array, size=3)

        # Calculate expected result manually using scipy's median_filter
        expected = median_filter(test_array, size=3, axes=[1, 2])

        # Check if results match
        np.testing.assert_array_equal(result, expected)

    def test_apply_median_filter_different_sizes(self):
        """Test median filter with different filter sizes."""
        # Create a random 3D array
        np.random.seed(42)  # For reproducibility
        test_array = np.random.randint(0, 100, size=(3, 5, 5))

        # Test with different filter sizes
        for size in [3, 5]:
            result = apply_median_filter(test_array, size=size)
            expected = median_filter(test_array, size=size, axes=[1, 2])
            np.testing.assert_array_equal(result, expected)

    def test_apply_median_filter_edge_case(self):
        """Test median filter with edge case (size=1)."""
        # Create a random 3D array
        np.random.seed(42)
        test_array = np.random.randint(0, 100, size=(2, 4, 4))

        # With size=1, the median filter should return the original array
        result = apply_median_filter(test_array, size=1)
        np.testing.assert_array_equal(result, test_array)

    def test_apply_median_filter_4d_array(self):
        """Test median filter on a 4D array."""
        # Create a 4D test array
        test_array = np.ones((2, 3, 3, 2))

        # Apply median filter
        result = apply_median_filter(test_array, size=3)

        # Calculate expected result
        expected = median_filter(test_array, size=3, axes=[1, 2])

        # Check if results match
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
