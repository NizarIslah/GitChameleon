import os
import sys
import unittest

import numpy as np

# Add the parent directory to the path so we can import the sample module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_280 import compute_fill_diagonal


class TestSample280(unittest.TestCase):
    
    def test_compute_fill_diagonal(self):
        # Create a test array
        test_array = np.ones((5, 5))
        radius = 1
        
        # Call the function (which modifies the array in place)
        compute_fill_diagonal(test_array, radius)
        
        # Expected result: diagonal and adjacent elements should remain 1,
        # other elements should be 0
        expected = np.array([
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1.],
            [0., 0., 0., 1., 1.]
        ])
        
        # Check if the modified test array matches the expected output
        np.testing.assert_array_equal(test_array, expected)
    
    def test_compute_fill_diagonal_with_larger_radius(self):
        # Create a test array
        test_array = np.ones((5, 5))
        radius = 2
        
        # Call the function (which modifies the array in place)
        compute_fill_diagonal(test_array, radius)
        
        # Expected result: diagonal and elements within radius 2 should remain 1,
        # other elements should be 0
        expected = np.array([
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1.]
        ])
        
        # Check if the modified test array matches the expected output
        np.testing.assert_array_equal(test_array, expected)
    
    def test_compute_fill_diagonal_with_zero_radius(self):
        # Create a test array
        test_array = np.ones((5, 5))
        radius = 0
        
        # Call the function (which modifies the array in place)
        compute_fill_diagonal(test_array, radius)
        
        # Expected result: only the diagonal elements should remain 1,
        # other elements should be 0
        expected = np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]
        ])
        
        # Check if the modified test array matches the expected output
        np.testing.assert_array_equal(test_array, expected)


if __name__ == '__main__':
    unittest.main()