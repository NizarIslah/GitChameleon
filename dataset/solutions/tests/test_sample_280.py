import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_280 import compute_fill_diagonal

class TestSample280(unittest.TestCase):
    
    def test_compute_fill_diagonal(self):
        test_array = np.ones((5, 5))
        radius = 1
        compute_fill_diagonal(test_array, radius)
        expected = np.array([
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1.],
            [0., 0., 0., 1., 1.]
        ])
        np.testing.assert_array_equal(test_array, expected)
    
    def test_compute_fill_diagonal_with_larger_radius(self):
        test_array = np.ones((5, 5))
        radius = 2
        compute_fill_diagonal(test_array, radius)
        expected = np.array([
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1.]
        ])
        np.testing.assert_array_equal(test_array, expected)
    
    def test_compute_fill_diagonal_with_zero_radius(self):
        test_array = np.ones((5, 5))
        radius = 0
        compute_fill_diagonal(test_array, radius)
        expected = np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]
        ])
        np.testing.assert_array_equal(test_array, expected)

if __name__ == '__main__':
    unittest.main()
