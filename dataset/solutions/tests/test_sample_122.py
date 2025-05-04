import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "solutions"))

from sample_122 import compute_determinant


class TestComputeDeterminant(unittest.TestCase):
    def test_2x2_matrix(self):
        """Test determinant calculation for a 2x2 matrix."""
        matrix = np.array([[4, 7], [2, 6]])
        expected = 10
        result = compute_determinant(matrix)
        self.assertAlmostEqual(result, expected, places=10)

    def test_3x3_matrix(self):
        """Test determinant calculation for a 3x3 matrix."""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = 0
        result = compute_determinant(matrix)
        self.assertEqual(result, expected)

    def test_identity_matrix(self):
        """Test determinant calculation for identity matrices."""
        for size in range(1, 5):
            matrix = np.eye(size)
            result = compute_determinant(matrix)
            self.assertEqual(result, 1.0)

    def test_singular_matrix(self):
        """Test determinant calculation for a singular matrix."""
        matrix = np.array([[1, 2], [2, 4]])
        expected = 0
        result = compute_determinant(matrix)
        self.assertAlmostEqual(result, expected, places=10)

    def test_float_values(self):
        """Test determinant calculation with floating point values."""
        matrix = np.array([[1.5, 2.5], [3.5, 4.5]])
        expected = -2.0
        result = compute_determinant(matrix)
        self.assertAlmostEqual(result, expected, places=10)


if __name__ == "__main__":
    unittest.main()
