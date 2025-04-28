import unittest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_140 import apply_maximum_filter

class TestApplyMaximumFilter(unittest.TestCase):
    
    def test_basic_functionality(self):
        """Test that the function works on a simple 2D array."""
        # Create a simple 2D array
        input_array = np.array([
            [1, 2, 3, 2, 1],
            [5, 6, 7, 6, 5]
        ])
        size = 3
        
        # Expected output: maximum_filter with size=3 applied to each row
        expected_output = np.array([
            [2, 3, 3, 3, 2],
            [6, 7, 7, 7, 6]
        ])
        
        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)
    
    def test_single_row(self):
        """Test with a single row array."""
        input_array = np.array([[1, 2, 3, 4, 5]])
        size = 3
        
        # Expected output: maximum_filter with size=3 applied to the row
        expected_output = np.array([[2, 3, 4, 5, 5]])
        
        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)
    
    def test_different_sizes(self):
        """Test with different filter sizes."""
        input_array = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ])
        
        # Test with size = 1 (should return the original array)
        result_size_1 = apply_maximum_filter(input_array, 1)
        np.testing.assert_array_equal(result_size_1, input_array)
        
        # Test with size = 5 (should return the max value for each row)
        result_size_5 = apply_maximum_filter(input_array, 5)
        expected_size_5 = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5]
        ])
        np.testing.assert_array_equal(result_size_5, expected_size_5)
    
    def test_with_zeros(self):
        """Test with an array containing zeros."""
        input_array = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0]
        ])
        size = 3
        
        expected_output = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ])
        
        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)
    
    def test_3d_array(self):
        """Test with a 3D array (multiple 2D arrays)."""
        input_array = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ])
        size = 2
        
        # Expected output: maximum_filter with size=2 applied to each 2D array
        expected_output = np.array([
            [[4, 4], [4, 4]],
            [[8, 8], [8, 8]]
        ])
        
        result = apply_maximum_filter(input_array, size)
        np.testing.assert_array_equal(result, expected_output)
    
    def test_empty_array(self):
        """Test with an empty array."""
        input_array = np.array([])
        size = 3
        
        result = apply_maximum_filter(input_array, size)
        self.assertEqual(result.size, 0)

if __name__ == '__main__':
    unittest.main()