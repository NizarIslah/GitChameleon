# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_129 import apply_rank_filter


class TestApplyRankFilter(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the function works with a simple 3D array."""
        # Create a 3D array (1, 3, 3) filled with values 0-8
        test_array = np.arange(9).reshape(1, 3, 3)

        # Apply rank filter with rank=0 (min value) and size=3
        result = apply_rank_filter(test_array, rank=0, size=3)

        # For a 3x3 window with rank=0, we should get the minimum value in each window
        # Since the filter is applied along axes 1 and 2, and the array is small,
        # the result should have the same shape as the input
        self.assertEqual(result.shape, test_array.shape)

        # Verify the function is correctly calling rank_filter
        # For this simple case, we can manually verify some expected values
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 8))

    def test_different_ranks(self):
        """Test the function with different rank values."""
        # Create a 3D array (2, 4, 4) with known values
        test_array = np.ones((2, 4, 4))
        test_array[0, 1:3, 1:3] = 5  # Set a region to a different value
        test_array[1, 0:2, 0:2] = 10  # Set another region to a different value

        # Test with minimum rank (0)
        min_result = apply_rank_filter(test_array, rank=0, size=3)
        # Test with maximum rank (size^2 - 1)
        max_result = apply_rank_filter(test_array, rank=8, size=3)
        # Test with median rank
        median_result = apply_rank_filter(test_array, rank=4, size=3)

        # Verify shapes
        self.assertEqual(min_result.shape, test_array.shape)
        self.assertEqual(max_result.shape, test_array.shape)
        self.assertEqual(median_result.shape, test_array.shape)

        # Min filter should have values <= original
        self.assertTrue(np.all(min_result <= test_array))
        # Max filter should have values >= original
        self.assertTrue(np.all(max_result >= test_array))

    def test_different_sizes(self):
        """Test the function with different filter sizes."""
        # Create a larger 3D array (2, 5, 5)
        test_array = np.ones((2, 5, 5))
        test_array[0, 2, 2] = 9  # Set center point to a different value

        # Test with size=1 (should return the original array)
        size1_result = apply_rank_filter(test_array, rank=0, size=1)
        # Test with size=3
        size3_result = apply_rank_filter(test_array, rank=4, size=3)
        # Test with size=5
        size5_result = apply_rank_filter(test_array, rank=12, size=5)

        # Verify shapes
        self.assertEqual(size1_result.shape, test_array.shape)
        self.assertEqual(size3_result.shape, test_array.shape)
        self.assertEqual(size5_result.shape, test_array.shape)

        # For size=1, the result should be the same as the input
        # (since a 1x1 window just contains the original value)
        np.testing.assert_array_almost_equal(size1_result, test_array)

    def test_edge_cases(self):
        """Test edge cases like empty arrays or extreme rank values."""
        # Test with a 3D array with a single element
        single_element = np.array([[[5]]])
        result = apply_rank_filter(single_element, rank=0, size=1)
        self.assertEqual(result.shape, single_element.shape)
        self.assertEqual(result[0, 0, 0], 5)

        # Test with a 4D array (should work since we're only filtering along axes 1 and 2)
        four_d_array = np.ones((2, 3, 3, 2))
        result = apply_rank_filter(four_d_array, rank=4, size=3)
        self.assertEqual(result.shape, four_d_array.shape)

        # Test with float values
        float_array = np.random.random((2, 4, 4))
        result = apply_rank_filter(float_array, rank=4, size=3)
        self.assertEqual(result.shape, float_array.shape)
        self.assertEqual(result.dtype, float_array.dtype)


if __name__ == "__main__":
    unittest.main()
