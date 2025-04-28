import unittest
import numpy as np
from scipy.ndimage import maximum_filter
from dataset.solutions.sample_139 import apply_maximum_filter

class TestMaximumFilter(unittest.TestCase):
    
    def test_apply_maximum_filter_basic(self):
        """Test basic functionality of apply_maximum_filter."""
        # Create a 3D array with shape (2, 3, 3)
        input_array = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            
            [[9, 8, 7],
             [6, 5, 4],
             [3, 2, 1]]
        ])
        
        # Apply maximum filter with size=3
        result = apply_maximum_filter(input_array, size=3)
        
        # Expected result: maximum value in each 3x3 window for axes 1 and 2
        # Since the filter size is 3 and our arrays are 3x3, each slice should
        # have the maximum value of that slice
        expected = np.array([
            [[9, 9, 9],
             [9, 9, 9],
             [9, 9, 9]],
            
            [[9, 9, 9],
             [9, 9, 9],
             [9, 9, 9]]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_apply_maximum_filter_different_size(self):
        """Test maximum filter with a different filter size."""
        # Create a 3D array with shape (2, 4, 4)
        input_array = np.array([
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]],
            
            [[16, 15, 14, 13],
             [12, 11, 10, 9],
             [8, 7, 6, 5],
             [4, 3, 2, 1]]
        ])
        
        # Apply maximum filter with size=2
        result = apply_maximum_filter(input_array, size=2)
        
        # Manually calculate expected result for size=2
        # For each 2x2 window, take the maximum value
        expected = np.array([
            [[6, 7, 8, 8],
             [10, 11, 12, 12],
             [14, 15, 16, 16],
             [14, 15, 16, 16]],
            
            [[16, 16, 15, 14],
             [16, 16, 15, 14],
             [12, 11, 10, 9],
             [8, 7, 6, 5]]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_apply_maximum_filter_with_zeros(self):
        """Test maximum filter with an array containing zeros."""
        # Create a 3D array with zeros
        input_array = np.zeros((2, 3, 3))
        input_array[0, 1, 1] = 5  # Set one value to non-zero
        
        # Apply maximum filter with size=2
        result = apply_maximum_filter(input_array, size=2)
        
        # Expected result
        expected = np.array([
            [[5, 5, 0],
             [5, 5, 0],
             [0, 0, 0]],
            
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_apply_maximum_filter_compare_with_direct(self):
        """Compare our function with direct call to scipy's maximum_filter."""
        # Create a random 3D array
        np.random.seed(42)  # For reproducibility
        input_array = np.random.rand(3, 5, 5)
        
        # Apply our function
        result = apply_maximum_filter(input_array, size=3)
        
        # Apply scipy's maximum_filter directly
        expected = maximum_filter(input_array, size=3, axes=[1, 2])
        
        # They should be identical
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()