import os
# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_280 import compute_fill_diagonal


class TestSample280(unittest.TestCase):
    
    def test_compute_fill_diagonal(self):
        # Create a test array
        test_array = np.ones((5, 5))
        radius = 1
        
        # Apply the function
        result = compute_fill_diagonal(test_array, radius)
        
        # Expected result: diagonal and adjacent elements should remain 1,
        # other elements should be 0
        expected = np.array([
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1.],
            [0., 0., 0., 1., 1.]
        ])
        
        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_fill_diagonal_with_larger_radius(self):
        # Create a test array
        test_array = np.ones((5, 5))
        radius = 2
        
        # Apply the function
        result = compute_fill_diagonal(test_array, radius)
        
        # Expected result: diagonal and elements within radius 2 should remain 1,
        # other elements should be 0
        expected = np.array([
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1.]
        ])
        
        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_fill_diagonal_with_zero_radius(self):
        # Create a test array
        test_array = np.ones((5, 5))
        radius = 0
        
        # Apply the function
        result = compute_fill_diagonal(test_array, radius)
        
        # Expected result: only the diagonal elements should remain 1,
        # other elements should be 0
        expected = np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]
        ])
        
        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()