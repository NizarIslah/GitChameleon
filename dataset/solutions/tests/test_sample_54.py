import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_54 import get_pairwise_dist


class TestManhattanDistance(unittest.TestCase):
    def test_get_pairwise_dist_simple(self):
        """Test simple Manhattan distances."""
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[1, 1], [2, 2]])
        # distances flattened: [2,4,0,2]
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, np.array([2., 4., 0., 2.]))

    def test_get_pairwise_dist_zero(self):
        """Test zero distances when X == Y."""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2], [3, 4]])
        # [0,4,4,0]
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, np.array([0., 4., 4., 0.]))

    def test_get_pairwise_dist_negative_values(self):
        """Test Manhattan distance with negative values."""
        X = np.array([[-1, -2], [3, 4]])
        Y = np.array([[1, 2], [-3, -4]])
        # [6,4,4,14]
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, np.array([6., 4., 4., 14.]))

    def test_get_pairwise_dist_empty_arrays(self):
        """Test that empty inputs raise a ValueError."""
        X = np.array([]).reshape(0, 0)
        Y = np.array([]).reshape(0, 0)
        with self.assertRaises(ValueError):
            get_pairwise_dist(X, Y)

    def test_get_pairwise_dist_single_point(self):
        """Test when Y has a single sample."""
        X = np.array([[0, 0], [1, 2]])
        Y = np.array([[3, 4]])
        # distances [7,4]
        result = get_pairwise_dist(X, Y)
        np.testing.assert_array_equal(result, np.array([7., 4.]))


if __name__ == "__main__":
    unittest.main()
