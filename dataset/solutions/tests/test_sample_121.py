import os
import sys
import unittest

import numpy as np
from scipy.linalg import det

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_121 import compute_determinant


class TestComputeDeterminant(unittest.TestCase):
    def test_single_matrix(self):
        """Test with a single matrix."""
        # Create a 2x2 matrix
        matrix = np.array([[1, 2], [3, 4]])
        # Reshape to have shape (1, 2, 2) to represent a batch with one matrix
        matrix = matrix.reshape(1, 2, 2)

        result = compute_determinant(matrix)

        # Expected determinant of [[1, 2], [3, 4]] is -2
        expected = np.array([-2.0])
        np.testing.assert_allclose(result, expected)

    def test_multiple_matrices(self):
        """Test with multiple matrices."""
        # Create a batch of 3 matrices
        matrices = np.array(
            [
                [[1, 2], [3, 4]],  # det = -2
                [[5, 6], [7, 8]],  # det = -2
                [[9, 10], [11, 12]],  # det = -2
            ]
        )

        result = compute_determinant(matrices)

        # Expected determinants
        expected = np.array([-2.0, -2.0, -2.0])
        np.testing.assert_allclose(result, expected)

    def test_3x3_matrices(self):
        """Test with 3x3 matrices."""
        # Create a batch of 2 3x3 matrices
        matrices = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # det = 0
                [[2, 0, 1], [0, 1, 0], [1, 0, 2]],  # det = 3
            ]
        )

        result = compute_determinant(matrices)

        # Expected determinants
        expected = np.array([0.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_empty_batch(self):
        """Test with an empty batch."""
        # Create an empty batch of 2x2 matrices
        matrices = np.zeros((0, 2, 2))

        result = compute_determinant(matrices)

        # Expected result is an empty array
        self.assertEqual(result.shape, (0,))

    def test_compare_with_scipy(self):
        """Test by comparing with scipy's det function directly."""
        # Create random matrices
        np.random.seed(42)  # For reproducibility
        matrices = np.random.rand(5, 4, 4)  # 5 random 4x4 matrices

        result = compute_determinant(matrices)

        # Calculate expected results using scipy directly
        expected = np.array([det(matrices[i]) for i in range(matrices.shape[0])])
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
