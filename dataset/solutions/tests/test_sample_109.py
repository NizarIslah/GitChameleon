# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.spatial import distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_109 import compute_wminkowski


class TestComputeWMinkowski(unittest.TestCase):
    def test_basic_case(self):
        """Test with simple vectors and weights."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        p = 2
        w = np.array([1, 1, 1])

        expected = distance.wminkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)

        self.assertEqual(result, expected)
        # The expected value should be the Euclidean distance when weights are all 1
        self.assertAlmostEqual(result, np.sqrt(27), places=10)

    def test_with_weights(self):
        """Test with non-uniform weights."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        p = 2
        w = np.array([0.5, 1.0, 2.0])

        expected = distance.wminkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)

        self.assertEqual(result, expected)
        # Manual calculation: sqrt((|0.5*(1-4)|^2 + |1.0*(2-5)|^2 + |2.0*(3-6)|^2))
        # = sqrt(2.25 + 9 + 36) = sqrt(47.25)
        self.assertAlmostEqual(result, np.sqrt(47.25), places=10)

    def test_different_p_value(self):
        """Test with a different p value (Manhattan distance)."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        p = 1
        w = np.array([1, 1, 1])

        expected = distance.wminkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)

        self.assertEqual(result, expected)
        # For p=1, this is the Manhattan distance: |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        self.assertEqual(result, 9.0)

    def test_higher_dimensions(self):
        """Test with higher dimensional vectors."""
        u = np.array([1, 2, 3, 4, 5])
        v = np.array([6, 7, 8, 9, 10])
        p = 2
        w = np.array([1, 1, 1, 1, 1])

        expected = distance.wminkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)

        self.assertEqual(result, expected)
        # The expected value should be sqrt(sum((v[i]-u[i])^2)) = sqrt(5*25) = sqrt(125)
        self.assertAlmostEqual(result, np.sqrt(125), places=10)

    def test_zero_distance(self):
        """Test with identical vectors (zero distance)."""
        u = np.array([1, 2, 3])
        v = np.array([1, 2, 3])
        p = 2
        w = np.array([1, 1, 1])

        expected = distance.wminkowski(u, v, p=p, w=w)
        result = compute_wminkowski(u, v, p, w)

        self.assertEqual(result, expected)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
