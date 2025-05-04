import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.ndimage import rank_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_130 import apply_rank_filter


class TestApplyRankFilter(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the function works with basic input."""
        # Create a simple 2D array
        test_array = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

        # Apply rank filter with rank=0 (min) and size=3
        result = apply_rank_filter(test_array, rank=0, size=3)

        # Manually calculate expected result
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = rank_filter(test_array[i], rank=0, size=3)

        # Check if results match
        np.testing.assert_array_equal(result, expected)

    def test_different_ranks(self):
        """Test with different rank values."""
        test_array = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

        # Test with rank=0 (min)
        min_result = apply_rank_filter(test_array, rank=0, size=3)

        # Test with rank=2 (median for size=3)
        med_result = apply_rank_filter(test_array, rank=1, size=3)

        # Test with rank=2 (max for size=3)
        max_result = apply_rank_filter(test_array, rank=2, size=3)

        # Verify results are different
        self.assertFalse(np.array_equal(min_result, med_result))
        self.assertFalse(np.array_equal(med_result, max_result))
        self.assertFalse(np.array_equal(min_result, max_result))

    def test_different_sizes(self):
        """Test with different filter sizes."""
        test_array = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

        # Test with size=3
        size3_result = apply_rank_filter(test_array, rank=1, size=3)

        # Test with size=5
        size5_result = apply_rank_filter(test_array, rank=2, size=5)

        # Verify results are different
        self.assertFalse(np.array_equal(size3_result, size5_result))

    def test_single_element_array(self):
        """Test with a single element array."""
        test_array = np.array([[7], [9]])

        # Apply rank filter
        result = apply_rank_filter(test_array, rank=0, size=1)

        # Expected result should be the same as input
        expected = np.array([[7], [9]])

        np.testing.assert_array_equal(result, expected)

    def test_3d_array(self):
        """Test with a 3D array."""
        test_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        # Reshape to 2D for our function
        reshaped = test_array.reshape(2, -1)

        # Apply rank filter
        result = apply_rank_filter(reshaped, rank=0, size=3)

        # Manually calculate expected result
        expected = np.zeros_like(reshaped)
        for i in range(reshaped.shape[0]):
            expected[i] = rank_filter(reshaped[i], rank=0, size=3)

        np.testing.assert_array_equal(result, expected)

    def test_random_array(self):
        """Test with a random array."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create a random array
        test_array = np.random.randint(0, 100, size=(5, 10))

        # Apply rank filter
        result = apply_rank_filter(test_array, rank=2, size=5)

        # Manually calculate expected result
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = rank_filter(test_array[i], rank=2, size=5)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
