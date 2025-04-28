import unittest
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_127 import apply_gaussian_filter1d

class TestGaussianFilter1D(unittest.TestCase):
    
    def test_apply_gaussian_filter1d_basic(self):
        """Test that the function returns expected output for a simple array."""
        # Create a simple input array
        x = np.array([1, 2, 3, 4, 5])
        radius = 2
        sigma = 1.0
        
        # Expected result using scipy's gaussian_filter1d directly
        expected = gaussian_filter1d(x, sigma=sigma)
        
        # Result from our function
        result = apply_gaussian_filter1d(x, radius=radius, sigma=sigma)
        
        # Check that the results are the same
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_apply_gaussian_filter1d_zeros(self):
        """Test with an array of zeros."""
        x = np.zeros(10)
        radius = 3
        sigma = 1.5
        
        # Expected result
        expected = gaussian_filter1d(x, sigma=sigma)
        
        # Result from our function
        result = apply_gaussian_filter1d(x, radius=radius, sigma=sigma)
        
        # Check that the results are the same
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_apply_gaussian_filter1d_random(self):
        """Test with a random array."""
        # Set a seed for reproducibility
        np.random.seed(42)
        
        # Create a random input array
        x = np.random.rand(20)
        radius = 4
        sigma = 2.0
        
        # Expected result
        expected = gaussian_filter1d(x, sigma=sigma)
        
        # Result from our function
        result = apply_gaussian_filter1d(x, radius=radius, sigma=sigma)
        
        # Check that the results are the same
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()