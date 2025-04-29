import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_73 import custom_cumproduct

class TestCustomCumproduct(unittest.TestCase):
    
    def test_custom_cumproduct_basic(self):
        """Test basic functionality with positive integers."""
        arr = np.array([1, 2, 3, 4])
        expected = np.array([1, 2, 6, 24])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)
    
    def test_custom_cumproduct_with_zeros(self):
        """Test with array containing zeros."""
        arr = np.array([2, 0, 3, 4])
        expected = np.array([2, 0, 0, 0])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)
    
    def test_custom_cumproduct_with_negative(self):
        """Test with array containing negative numbers."""
        arr = np.array([1, -2, 3, -4])
        expected = np.array([1, -2, -6, 24])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)
    
    def test_custom_cumproduct_empty_array(self):
        """Test with empty array."""
        arr = np.array([])
        expected = np.array([])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)
    
    def test_custom_cumproduct_single_element(self):
        """Test with single element array."""
        arr = np.array([5])
        expected = np.array([5])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)
    
    def test_custom_cumproduct_float_values(self):
        """Test with floating point values."""
        arr = np.array([1.5, 2.5, 0.5])
        expected = np.array([1.5, 3.75, 1.875])
        np.testing.assert_almost_equal(custom_cumproduct(arr), expected)
    
    def test_custom_cumproduct_2d_array(self):
        """Test with 2D array."""
        arr = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 2], [3, 8]])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)

if __name__ == '__main__':
    unittest.main()