import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_53 import get_pairwise_dist


class TestSample53(unittest.TestCase):
    def _summed_distances(self, X, Y):
        """Helper to reshape the flattened pairwise distances and sum per X sample."""
        flat = get_pairwise_dist(X, Y)
        # reshape to (n_X, n_Y) and sum along axis 1
        return flat.reshape(X.shape[0], Y.shape[0]).sum(axis=1)

    def test_get_pairwise_dist_simple_case(self):
        # Simple test case with 2D arrays
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[1, 1], [2, 2]])

        expected = np.array([6, 2])
        result = self._summed_distances(X, Y)

        np.testing.assert_array_equal(result, expected)

    def test_get_pairwise_dist_zero_distance(self):
        # Test with identical arrays
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2], [3, 4]])

        expected = np.array([4, 4])
        result = self._summed_distances(X, Y)

        np.testing.assert_array_equal(result, expected)

    def test_get_pairwise_dist_negative_values(self):
        # Test with negative values
        X = np.array([[-1, -2], [3, 4]])
        Y = np.array([[1, 2], [-3, -4]])

        expected = np.array([10, 18])
        result = self._summed_distances(X, Y)

        np.testing.assert_array_equal(result, expected)

    def test_get_pairwise_dist_single_point(self):
        # When Y has only one point, the distance is just that one distance
        X = np.array([[0, 0], [1, 2]])
        Y = np.array([[3, 4]])
        # distances: [|0-3|+|0-4|=7, |1-3|+|2-4|=4]
        expected = np.array([7, 4])
        result = self._summed_distances(X, Y)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
