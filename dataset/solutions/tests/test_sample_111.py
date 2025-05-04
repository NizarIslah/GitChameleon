import os
import sys
import unittest

import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_111 import compute_matrix_exponential


class TestMatrixExponential(unittest.TestCase):
    def test_single_matrix(self):
        # Test with a single matrix (wrapped in a batch dimension)
        A = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # Identity matrix
        result = compute_matrix_exponential(A)

        # Expected: e^I = e * I
        expected = np.array([[[np.e, 0.0], [0.0, np.e]]])
        np.testing.assert_allclose(result, expected, rtol=1e-7)

        # Check shape
        self.assertEqual(result.shape, A.shape)

    def test_multiple_matrices(self):
        # Test with multiple matrices
        A = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],  # Zero matrix
                [[1.0, 0.0], [0.0, 1.0]],  # Identity matrix
                [[0.0, 1.0], [0.0, 0.0]],  # Nilpotent matrix
            ]
        )

        result = compute_matrix_exponential(A)

        # Expected results:
        # e^0 = I
        # e^I = e * I
        # e^N = I + N (for this specific nilpotent matrix)
        expected = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[np.e, 0.0], [0.0, np.e]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        )

        np.testing.assert_allclose(result, expected, rtol=1e-7)
        self.assertEqual(result.shape, A.shape)

    def test_diagonal_matrix(self):
        # Test with a diagonal matrix
        A = np.array([[[2.0, 0.0], [0.0, 3.0]]])
        result = compute_matrix_exponential(A)

        # Expected: diagonal elements are e^values
        expected = np.array([[[np.exp(2.0), 0.0], [0.0, np.exp(3.0)]]])
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_compare_with_scipy_direct(self):
        # Test by comparing with direct scipy implementation
        matrices = np.random.rand(5, 3, 3)  # 5 random 3x3 matrices

        # Our implementation
        result = compute_matrix_exponential(matrices)

        # Direct scipy implementation for comparison
        expected = np.stack(
            [linalg.expm(matrices[i]) for i in range(matrices.shape[0])], axis=0
        )

        np.testing.assert_allclose(result, expected, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
