import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_54 import get_pairwise_dist


class TestManhattanDistance(unittest.TestCase):
    def test_get_pairwise_dist_simple(self):
        """Test simple Manhattan distances for 2x2 arrays."""
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[1, 1], [2, 2]])
        # distances:
        # [[|0-1|+|0-1|, |0-2|+|0-2|],
        #  [|1-1|+|1-1|, |1-2|+|1-2|]]
        # = [[2, 4],
        #    [0, 2]]
        result = get_pairwise_dist(X, Y)
        expected = np.array([[2.0, 4.0], [0.0, 2.0]])
        np.testing.assert_array_equal(result, expected)

    def test_get_pairwise_dist_zero(self):
        """Test zero distances when X == Y."""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2], [3, 4]])
        # distances:
        # [[0, 4],
        #  [4, 0]]
        result = get_pairwise_dist(X, Y)
        expected = np.array([[0.0, 4.0], [4.0, 0.0]])
        np.testing.assert_array_equal(result, expected)

    def test_get_pairwise_dist_negative_values(self):
        """Test Manhattan distance with negative values."""
        X = np.array([[-1, -2], [3, 4]])
        Y = np.array([[1, 2], [-3, -4]])
        # distances:
        # [[6, 4],
        #  [4, 14]]
        result = get_pairwise_dist(X, Y)
        expected = np.array([[6.0, 4.0], [4.0, 14.0]])
        np.testing.assert_array_equal(result, expected)

    def test_get_pairwise_dist_single_point(self):
        """Test when Y has a single sample (shape Nx1)."""
        X = np.array([[0, 0], [1, 2]])
        Y = np.array([[3, 4]])
        # distances:
        # [[7],
        #  [4]]
        result = get_pairwise_dist(X, Y)
        expected = np.array([[7.0], [4.0]])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
