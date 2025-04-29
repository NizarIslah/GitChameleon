import unittest
import numpy as np
from dataset.solutions.sample_54 import get_pairwise_dist

class TestManhattanDistance(unittest.TestCase):
    
    def test_get_pairwise_dist_same_arrays(self):
        """Test with identical arrays."""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2], [3, 4]])
        result = get_pairwise_dist(X, Y)
        expected = np.array([[0, 4], [4, 0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_get_pairwise_dist_different_arrays(self):
        """Test with different arrays."""
        X = np.array([[0, 1], [2, 3]])
        Y = np.array([[5, 6], [7, 8]])
        result = get_pairwise_dist(X, Y)
        expected = np.array([[10, 14], [6, 10]])
        np.testing.assert_array_equal(result, expected)
    
    def test_get_pairwise_dist_different_shapes(self):
        """Test with arrays of different shapes."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        Y = np.array([[7, 8]])
        result = get_pairwise_dist(X, Y)
        expected = np.array([[12], [8], [4]])
        np.testing.assert_array_equal(result, expected)
    
    def test_get_pairwise_dist_empty_arrays(self):
        """Test with empty arrays."""
        X = np.array([])
        Y = np.array([])
        # Reshape to 2D arrays with 0 rows and 0 columns
        X = X.reshape(0, 0)
        Y = Y.reshape(0, 0)
        result = get_pairwise_dist(X, Y)
        expected = np.array([]).reshape(0, 0)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()