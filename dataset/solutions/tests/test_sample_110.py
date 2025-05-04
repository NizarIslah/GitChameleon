# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.spatial import distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_110 import compute_wminkowski


class TestComputeWMinkowski(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that compute_wminkowski correctly calculates weighted Minkowski distance."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        p = 2
        w = np.array([1, 1, 1])

        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)

        self.assertEqual(result, expected)

    def test_different_p_values(self):
        """Test with different p values (Manhattan, Euclidean, etc.)."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        w = np.array([1, 1, 1])

        # Test with p=1 (Manhattan distance)
        p = 1
        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)

        # Test with p=3
        p = 3
        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)

    def test_different_weights(self):
        """Test with different weight values."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        p = 2

        # Test with non-uniform weights
        w = np.array([0.5, 1.0, 2.0])
        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)

        # Test with zero weights (should handle this case)
        w = np.array([0, 1, 1])
        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)

    def test_different_dimensions(self):
        """Test with arrays of different dimensions."""
        # 2D arrays
        u = np.array([1, 2])
        v = np.array([4, 5])
        p = 2
        w = np.array([1, 1])

        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)

        # 4D arrays
        u = np.array([1, 2, 3, 4])
        v = np.array([5, 6, 7, 8])
        w = np.array([1, 1, 1, 1])

        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)

    def test_edge_cases(self):
        """Test edge cases like identical vectors."""
        # Identical vectors should have distance 0
        u = np.array([1, 2, 3])
        v = np.array([1, 2, 3])
        p = 2
        w = np.array([1, 1, 1])

        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, 0.0)

        # Very large p value
        p = 10
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        expected = distance.minkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
