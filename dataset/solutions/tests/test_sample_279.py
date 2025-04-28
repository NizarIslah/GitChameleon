import os
# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_279 import compute_fill_diagonal


class TestComputeFillDiagonal(unittest.TestCase):
    
    def test_square_matrix(self):
        """Test with a square matrix and different radius values."""
        # Create a 5x5 matrix filled with ones
        x = np.ones((5, 5))
        
        # Test with radius 0 (only diagonal should remain)
        result = compute_fill_diagonal(x.copy(), 0)
        expected = np.eye(5)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with radius 1 (diagonal and one off-diagonal should remain)
        result = compute_fill_diagonal(x.copy(), 1)
        expected = np.ones((5, 5))
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 1:
                    expected[i, j] = 0
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with radius 2
        result = compute_fill_diagonal(x.copy(), 2)
        expected = np.ones((5, 5))
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 2:
                    expected[i, j] = 0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rectangular_matrix(self):
        """Test with a rectangular matrix."""
        # Create a 3x4 matrix filled with ones
        x = np.ones((3, 4))
        
        # Test with radius 1
        result = compute_fill_diagonal(x.copy(), 1)
        expected = np.ones((3, 4))
        for i in range(3):
            for j in range(4):
                if abs(i - j) > 1:
                    expected[i, j] = 0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_large_radius(self):
        """Test with a radius larger than matrix dimensions."""
        # Create a 3x3 matrix
        x = np.ones((3, 3))
        
        # Test with radius 10 (should keep all elements)
        result = compute_fill_diagonal(x.copy(), 10)
        np.testing.assert_array_almost_equal(result, x)
    
    def test_zero_matrix(self):
        """Test with a matrix of zeros."""
        x = np.zeros((4, 4))
        result = compute_fill_diagonal(x.copy(), 1)
        np.testing.assert_array_almost_equal(result, x)
    
    def test_identity_matrix(self):
        """Test with an identity matrix."""
        x = np.eye(4)
        
        # With radius 0, should remain unchanged
        result = compute_fill_diagonal(x.copy(), 0)
        np.testing.assert_array_almost_equal(result, x)
        
        # With radius 1, should remain unchanged
        result = compute_fill_diagonal(x.copy(), 1)
        expected = np.eye(4)
        for i in range(4):
            for j in range(4):
                if abs(i - j) == 1:
                    expected[i, j] = 1
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    unittest.main()