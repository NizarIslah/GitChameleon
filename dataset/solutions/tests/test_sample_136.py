import unittest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_136 import apply_uniform_filter

class TestUniformFilter(unittest.TestCase):
    
    def test_apply_uniform_filter_1d(self):
        """Test uniform filter on a batch of 1D arrays"""
        # Create a batch of 1D arrays
        input_array = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ])
        size = 3
        
        result = apply_uniform_filter(input_array, size)
        
        # Check shape is preserved
        self.assertEqual(result.shape, input_array.shape)
        
        # For uniform filter with size=3, the edge values will be affected by padding
        # We can check the middle values which should be averages of the surrounding values
        # For the first row, the middle value should be approximately (2+3+4)/3 = 3
        # For the second row, the middle value should be approximately (4+3+2)/3 = 3
        self.assertAlmostEqual(result[0, 2], 3.0, places=5)
        self.assertAlmostEqual(result[1, 2], 3.0, places=5)
    
    def test_apply_uniform_filter_2d(self):
        """Test uniform filter on a batch of 2D arrays"""
        # Create a batch of 2D arrays (2 images of 3x3)
        input_array = np.array([
            [[1, 2, 3], 
             [4, 5, 6], 
             [7, 8, 9]],
            
            [[9, 8, 7], 
             [6, 5, 4], 
             [3, 2, 1]]
        ])
        size = 2
        
        result = apply_uniform_filter(input_array, size)
        
        # Check shape is preserved
        self.assertEqual(result.shape, input_array.shape)
        
        # For a 2D uniform filter with size=2, each output pixel is the average of a 2x2 neighborhood
        # We can check some values to ensure the filter is applied correctly
        # For example, for the first image, the value at position (1,1) should be influenced by its neighbors
        self.assertTrue(3.0 < result[0, 1, 1] < 7.0)  # Should be around 5 (average of the center region)
        self.assertTrue(3.0 < result[1, 1, 1] < 7.0)  # Should be around 5 for the second image too
    
    def test_apply_uniform_filter_empty(self):
        """Test uniform filter on an empty array"""
        # Create an empty batch
        input_array = np.zeros((0, 5))
        size = 3
        
        result = apply_uniform_filter(input_array, size)
        
        # Check shape is preserved
        self.assertEqual(result.shape, input_array.shape)
    
    def test_apply_uniform_filter_size_1(self):
        """Test uniform filter with size=1 (should return the original array)"""
        input_array = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        size = 1
        
        result = apply_uniform_filter(input_array, size)
        
        # With size=1, the result should be very close to the original array
        np.testing.assert_allclose(result, input_array, rtol=1e-5)
    
    def test_apply_uniform_filter_3d(self):
        """Test uniform filter on a batch of 3D arrays"""
        # Create a batch of 3D arrays (2 volumes of 2x2x2)
        input_array = np.array([
            [[[1, 2], [3, 4]], 
             [[5, 6], [7, 8]]],
            
            [[[8, 7], [6, 5]], 
             [[4, 3], [2, 1]]]
        ])
        size = 2
        
        result = apply_uniform_filter(input_array, size)
        
        # Check shape is preserved
        self.assertEqual(result.shape, input_array.shape)
        
        # For 3D data, we just verify the function runs and returns the expected shape

if __name__ == '__main__':
    unittest.main()