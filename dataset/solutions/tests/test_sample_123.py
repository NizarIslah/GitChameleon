import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.linalg import lu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_123 import compute_lu_decomposition


class TestLUDecomposition(unittest.TestCase):
    def test_compute_lu_decomposition_single_matrix(self):
        """Test LU decomposition on a single matrix (1x3x3)."""
        # Create a test matrix
        A = np.array([[4, 3, 2], [2, 1, 3], [1, 4, 5]])
        A = A.reshape(1, 3, 3)  # Make it a 1x3x3 array

        # Compute LU decomposition using our function
        p, l, u = compute_lu_decomposition(A)

        # Verify shapes
        self.assertEqual(p.shape, A.shape)
        self.assertEqual(l.shape, A.shape)
        self.assertEqual(u.shape, A.shape)

        # Verify the decomposition by reconstructing the original matrix
        # For each matrix in the batch
        for i in range(A.shape[0]):
            # Permutation matrix times L times U should equal the original matrix
            reconstructed = np.dot(np.dot(p[i], l[i]), u[i])
            np.testing.assert_allclose(reconstructed, A[i], rtol=1e-5)

    def test_compute_lu_decomposition_multiple_matrices(self):
        """Test LU decomposition on multiple matrices (3x2x2)."""
        # Create a batch of test matrices
        A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

        # Compute LU decomposition using our function
        p, l, u = compute_lu_decomposition(A)

        # Verify shapes
        self.assertEqual(p.shape, A.shape)
        self.assertEqual(l.shape, A.shape)
        self.assertEqual(u.shape, A.shape)

        # Verify the decomposition by reconstructing the original matrix
        # For each matrix in the batch
        for i in range(A.shape[0]):
            # Permutation matrix times L times U should equal the original matrix
            reconstructed = np.dot(np.dot(p[i], l[i]), u[i])
            np.testing.assert_allclose(reconstructed, A[i], rtol=1e-5)

    def test_compute_lu_decomposition_empty_array(self):
        """Test LU decomposition on an empty array."""
        # Create an empty array with the right shape
        A = np.zeros((0, 2, 2))

        # Compute LU decomposition using our function
        p, l, u = compute_lu_decomposition(A)

        # Verify shapes
        self.assertEqual(p.shape, A.shape)
        self.assertEqual(l.shape, A.shape)
        self.assertEqual(u.shape, A.shape)

    def test_compare_with_scipy_lu(self):
        """Test that our function gives the same results as scipy's lu function."""
        # Create a test matrix
        A = np.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
        )

        # Compute LU decomposition using our function
        p_our, l_our, u_our = compute_lu_decomposition(A)

        # Compute LU decomposition using scipy's lu function directly
        p_scipy = np.zeros(A.shape)
        l_scipy = np.zeros(A.shape)
        u_scipy = np.zeros(A.shape)

        for i in range(A.shape[0]):
            p_scipy[i], l_scipy[i], u_scipy[i] = lu(A[i])

        # Verify that our results match scipy's results
        np.testing.assert_allclose(p_our, p_scipy)
        np.testing.assert_allclose(l_our, l_scipy)
        np.testing.assert_allclose(u_our, u_scipy)


if __name__ == "__main__":
    unittest.main()
