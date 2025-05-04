import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_77 import custom_product


class TestCustomProduct(unittest.TestCase):
    def test_custom_product_with_positive_numbers(self):
        """Test custom_product with an array of positive numbers."""
        arr = np.array([1, 2, 3, 4, 5])
        result = custom_product(arr)
        expected = 120  # 1*2*3*4*5 = 120
        self.assertEqual(result, expected)

    def test_custom_product_with_negative_numbers(self):
        """Test custom_product with an array containing negative numbers."""
        arr = np.array([-1, 2, -3, 4])
        result = custom_product(arr)
        expected = 24  # -1*2*-3*4 = 24
        self.assertEqual(result, expected)

    def test_custom_product_with_zeros(self):
        """Test custom_product with an array containing zeros."""
        arr = np.array([1, 0, 3, 4])
        result = custom_product(arr)
        expected = 0  # Any product with 0 should be 0
        self.assertEqual(result, expected)

    def test_custom_product_with_empty_array(self):
        """Test custom_product with an empty array."""
        arr = np.array([])
        result = custom_product(arr)
        # NumPy's product of an empty array is 1.0 by default
        expected = 1.0
        self.assertEqual(result, expected)

    def test_custom_product_with_single_element(self):
        """Test custom_product with a single element array."""
        arr = np.array([42])
        result = custom_product(arr)
        expected = 42
        self.assertEqual(result, expected)

    def test_custom_product_with_float_values(self):
        """Test custom_product with floating point values."""
        arr = np.array([1.5, 2.5, 3.0])
        result = custom_product(arr)
        expected = 11.25  # 1.5*2.5*3.0 = 11.25
        self.assertAlmostEqual(result, expected, places=10)

    def test_custom_product_with_large_numbers(self):
        """Test custom_product with large numbers."""
        arr = np.array([1e5, 1e5, 1e5])
        result = custom_product(arr)
        expected = 1e15  # 1e5*1e5*1e5 = 1e15
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
