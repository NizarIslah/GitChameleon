import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_53 import get_pairwise_dist


class TestSample53(unittest.TestCase):
    def test_get_pairwise_dist_simple_case(self):
        # Simple test case with 2D arrays
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[1, 1], [2, 2]])
        
        # Expected result:
        # Manhattan distance between [0,0] and [1,1] = |0-1| + |0-1| = 2
        # Manhattan distance between [0,0] and [2,2] = |0-2| + |0-2| = 4
        # Manhattan distance between [1,1] and [1,1] = |1-1| + |1-1| = 0
        # Manhattan distance between [1,1] and [2,2] = |1-2| + |1-2| = 2
        # Sum over features for each X sample:
        # For [0,0]: 2 + 4 = 6
        # For [1,1]: 0 + 2 = 2
        expected = np.array([6, 2])
        
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, expected)
    
    def test_get_pairwise_dist_different_dimensions(self):
        # Test with different dimensions
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[7, 8, 9]])
        
        # Expected result:
        # Manhattan distance between [1,2,3] and [7,8,9] = |1-7| + |2-8| + |3-9| = 6 + 6 + 6 = 18
        # Manhattan distance between [4,5,6] and [7,8,9] = |4-7| + |5-8| + |6-9| = 3 + 3 + 3 = 9
        # Sum over features for each X sample:
        # For [1,2,3]: 18
        # For [4,5,6]: 9
        expected = np.array([18, 9])
        
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, expected)
    
    def test_get_pairwise_dist_zero_distance(self):
        # Test with identical arrays
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2], [3, 4]])
        
        # Expected result:
        # Manhattan distance between [1,2] and [1,2] = |1-1| + |2-2| = 0
        # Manhattan distance between [1,2] and [3,4] = |1-3| + |2-4| = 2 + 2 = 4
        # Manhattan distance between [3,4] and [1,2] = |3-1| + |4-2| = 2 + 2 = 4
        # Manhattan distance between [3,4] and [3,4] = |3-3| + |4-4| = 0
        # Sum over features for each X sample:
        # For [1,2]: 0 + 4 = 4
        # For [3,4]: 4 + 0 = 4
        expected = np.array([4, 4])
        
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, expected)
    
    def test_get_pairwise_dist_negative_values(self):
        # Test with negative values
        X = np.array([[-1, -2], [3, 4]])
        Y = np.array([[1, 2], [-3, -4]])
        
        # Expected result:
        # Manhattan distance between [-1,-2] and [1,2] = |-1-1| + |-2-2| = 2 + 4 = 6
        # Manhattan distance between [-1,-2] and [-3,-4] = |-1-(-3)| + |-2-(-4)| = 2 + 2 = 4
        # Manhattan distance between [3,4] and [1,2] = |3-1| + |4-2| = 2 + 2 = 4
        # Manhattan distance between [3,4] and [-3,-4] = |3-(-3)| + |4-(-4)| = 6 + 8 = 14
        # Sum over features for each X sample:
        # For [-1,-2]: 6 + 4 = 10
        # For [3,4]: 4 + 14 = 18
        expected = np.array([10, 18])
        
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()