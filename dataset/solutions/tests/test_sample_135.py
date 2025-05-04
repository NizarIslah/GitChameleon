# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_135 import apply_uniform_filter


class TestUniformFilter(unittest.TestCase):
    def test_apply_uniform_filter_3d_array(self):
        """Test uniform filter on a 3D array."""
        # Create a 3D test array
        test_array = np.ones((3, 4, 5))
        size = 3

        # Apply our function
        result = apply_uniform_filter(test_array, size)

        # Apply the filter directly for comparison
        expected = uniform_filter(test_array, size=size, axes=[1, 2])

        # Check if results match
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_uniform_filter_different_sizes(self):
        """Test uniform filter with different filter sizes."""
        # Create a 3D test array with random values
        np.random.seed(42)  # For reproducibility
        test_array = np.random.rand(2, 6, 6)

        # Test with different filter sizes
        for size in [2, 3, 5]:
            result = apply_uniform_filter(test_array, size)
            expected = uniform_filter(test_array, size=size, axes=[1, 2])
            np.testing.assert_array_almost_equal(result, expected)

    def test_apply_uniform_filter_edge_cases(self):
        """Test uniform filter with edge cases."""
        # Test with size=1 (should return the original array)
        test_array = np.random.rand(2, 3, 3)
        result = apply_uniform_filter(test_array, 1)
        np.testing.assert_array_almost_equal(result, test_array)

        # Test with a 4D array
        test_array_4d = np.ones((2, 3, 4, 5))
        size = 2
        result = apply_uniform_filter(test_array_4d, size)
        expected = uniform_filter(test_array_4d, size=size, axes=[1, 2])
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
