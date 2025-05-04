import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_72 import custom_product


class TestCustomProduct(unittest.TestCase):
    def test_custom_product_with_positive_numbers(self):
        # Test with positive numbers
        arr = np.array([2, 3, 4])
        result = custom_product(arr)
        self.assertEqual(result, 24)

    def test_custom_product_with_negative_numbers(self):
        # Test with negative numbers
        arr = np.array([-2, 3, -4])
        result = custom_product(arr)
        self.assertEqual(result, 24)

    def test_custom_product_with_zeros(self):
        # Test with array containing zero
        arr = np.array([2, 0, 4])
        result = custom_product(arr)
        self.assertEqual(result, 0)

    def test_custom_product_with_single_element(self):
        # Test with single element array
        arr = np.array([5])
        result = custom_product(arr)
        self.assertEqual(result, 5)

    def test_custom_product_with_empty_array(self):
        # Test with empty array
        # NumPy's prod returns 1 for empty arrays (product identity element)
        arr = np.array([])
        result = custom_product(arr)
        self.assertEqual(result, 1)

    def test_custom_product_with_float_values(self):
        # Test with float values
        arr = np.array([1.5, 2.5])
        result = custom_product(arr)
        self.assertAlmostEqual(result, 3.75)

    def test_custom_product_with_multidimensional_array(self):
        # Test with multidimensional array
        # NumPy's prod flattens the array by default
        arr = np.array([[1, 2], [3, 4]])
        result = custom_product(arr)
        self.assertEqual(result, 24)


if __name__ == "__main__":
    unittest.main()
