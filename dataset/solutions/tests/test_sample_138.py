import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.ndimage import minimum_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_138 import apply_minimum_filter


class TestMinimumFilter(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the function works with a simple array."""
        # Create a test array
        test_array = np.array([[5, 2, 3, 1, 4], [7, 6, 9, 8, 5]])

        # Apply filter with size 3
        result = apply_minimum_filter(test_array, 3)

        # Expected result: manually calculate minimum filter for each row
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = minimum_filter(test_array[i], size=3)

        # Check if results match
        np.testing.assert_array_equal(result, expected)

    def test_different_sizes(self):
        """Test with different filter sizes."""
        # Create a test array
        test_array = np.array([[10, 8, 6, 4, 2], [1, 3, 5, 7, 9]])

        # Test with size 1 (should return the original array)
        result_size_1 = apply_minimum_filter(test_array, 1)
        np.testing.assert_array_equal(result_size_1, test_array)

        # Test with size 2
        result_size_2 = apply_minimum_filter(test_array, 2)
        expected_size_2 = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected_size_2[i] = minimum_filter(test_array[i], size=2)
        np.testing.assert_array_equal(result_size_2, expected_size_2)

        # Test with size 5 (entire row)
        result_size_5 = apply_minimum_filter(test_array, 5)
        expected_size_5 = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected_size_5[i] = minimum_filter(test_array[i], size=5)
        np.testing.assert_array_equal(result_size_5, expected_size_5)

    def test_empty_array(self):
        """Test with an empty array."""
        test_array = np.array([])
        # Reshape to 2D with 0 rows and 0 columns
        test_array = test_array.reshape(0, 0)

        # This should return an empty array without errors
        result = apply_minimum_filter(test_array, 3)
        self.assertEqual(result.size, 0)

    def test_single_element(self):
        """Test with a single element array."""
        test_array = np.array([[7]])
        result = apply_minimum_filter(test_array, 1)
        np.testing.assert_array_equal(result, test_array)

    def test_large_array(self):
        """Test with a larger random array."""
        # Create a random array
        np.random.seed(42)  # For reproducibility
        test_array = np.random.randint(0, 100, size=(5, 10))

        # Apply filter with size 3
        result = apply_minimum_filter(test_array, 3)

        # Calculate expected result
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = minimum_filter(test_array[i], size=3)

        # Check if results match
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
