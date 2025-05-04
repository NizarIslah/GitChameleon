import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.ndimage import median_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_134 import apply_median_filter


class TestMedianFilter(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the function applies median filter correctly to a simple array."""
        # Create a test array with some noise
        test_array = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])

        # Apply our function with filter size 3
        result = apply_median_filter(test_array, 3)

        # Calculate expected result manually for comparison
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = median_filter(test_array[i], size=3)

        # Check if results match
        np.testing.assert_array_equal(result, expected)

    def test_different_filter_sizes(self):
        """Test the function with different filter sizes."""
        test_array = np.array([[1, 10, 3, 20, 5], [5, 30, 3, 40, 1]])

        # Test with filter size 3
        result_size3 = apply_median_filter(test_array, 3)
        expected_size3 = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected_size3[i] = median_filter(test_array[i], size=3)
        np.testing.assert_array_equal(result_size3, expected_size3)

        # Test with filter size 5
        result_size5 = apply_median_filter(test_array, 5)
        expected_size5 = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected_size5[i] = median_filter(test_array[i], size=5)
        np.testing.assert_array_equal(result_size5, expected_size5)

    def test_shape_preservation(self):
        """Test that the function preserves the shape of the input array."""
        # Test with arrays of different shapes
        shapes = [(3, 5), (10, 2), (1, 20), (7, 7)]

        for shape in shapes:
            test_array = np.random.randint(0, 100, size=shape)
            result = apply_median_filter(test_array, 3)
            self.assertEqual(result.shape, test_array.shape)

    def test_single_row(self):
        """Test with a single row array."""
        test_array = np.array([[1, 9, 2, 8, 3, 7, 4, 6, 5]])
        result = apply_median_filter(test_array, 3)

        expected = np.zeros_like(test_array)
        expected[0] = median_filter(test_array[0], size=3)

        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test with an empty array (0 rows)."""
        test_array = np.zeros((0, 5))
        result = apply_median_filter(test_array, 3)

        # Should return an empty array with the same shape
        self.assertEqual(result.shape, test_array.shape)
        self.assertEqual(result.size, 0)

    def test_large_filter_size(self):
        """Test with a filter size larger than the array width."""
        test_array = np.array([[1, 2, 3], [4, 5, 6]])

        result = apply_median_filter(test_array, 5)

        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = median_filter(test_array[i], size=5)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
