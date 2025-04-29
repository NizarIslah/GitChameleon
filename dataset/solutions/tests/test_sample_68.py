import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_68 import apply_correlate_full

class TestApplyCorrelateFull(unittest.TestCase):
    
    def test_basic_correlation(self):
        """Test basic correlation between two arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([0, 1, 0.5])
        result = apply_correlate_full(arr1, arr2)
        
        # Calculate expected result manually
        expected = np.correlate(arr1, arr2, mode="full")
        
        np.testing.assert_array_almost_equal(result, expected)
        # The result should be [0.5, 2.0, 3.5, 3.0, 0.0]
        np.testing.assert_array_almost_equal(result, np.array([0.5, 2.0, 3.5, 3.0, 0.0]))
    
    def test_empty_arrays(self):
        """Test correlation with empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([])
        
        # Empty arrays should return empty result
        result = apply_correlate_full(arr1, arr2)
        self.assertEqual(result.size, 0)
    
    def test_different_length_arrays(self):
        """Test correlation with arrays of different lengths."""
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([0, 1])
        result = apply_correlate_full(arr1, arr2)
        
        # Calculate expected result
        expected = np.correlate(arr1, arr2, mode="full")
        
        np.testing.assert_array_almost_equal(result, expected)
        # The result should be [0, 1, 2, 3, 4, 0]
        np.testing.assert_array_almost_equal(result, np.array([0, 1, 2, 3, 4, 0]))
    
    def test_same_arrays(self):
        """Test correlation of an array with itself."""
        arr = np.array([1, 2, 3])
        result = apply_correlate_full(arr, arr)
        
        # Calculate expected result
        expected = np.correlate(arr, arr, mode="full")
        
        np.testing.assert_array_almost_equal(result, expected)
        # The result should be [3, 8, 14, 8, 3]
        np.testing.assert_array_almost_equal(result, np.array([3, 8, 14, 8, 3]))
    
    def test_with_negative_values(self):
        """Test correlation with arrays containing negative values."""
        arr1 = np.array([1, -2, 3])
        arr2 = np.array([-1, 0, 1])
        result = apply_correlate_full(arr1, arr2)
        
        # Calculate expected result
        expected = np.correlate(arr1, arr2, mode="full")
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_with_float_values(self):
        """Test correlation with arrays containing float values."""
        arr1 = np.array([1.5, 2.5, 3.5])
        arr2 = np.array([0.5, 1.5])
        result = apply_correlate_full(arr1, arr2)
        
        # Calculate expected result
        expected = np.correlate(arr1, arr2, mode="full")
        
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()