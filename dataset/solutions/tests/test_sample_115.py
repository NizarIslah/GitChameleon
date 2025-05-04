import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy import linalg, sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_115 import compute_matrix_exponential


class TestMatrixExponential(unittest.TestCase):
    def test_identity_matrix(self):
        """Test that exp(0) = I, where 0 is a zero matrix and I is an identity matrix."""
        # Create a sparse zero matrix
        n = 3
        zero_matrix = sparse.lil_matrix((n, n))

        # Compute the exponential
        result = compute_matrix_exponential(zero_matrix)

        # The exponential of a zero matrix should be an identity matrix
        expected = sparse.eye(n).toarray()
        np.testing.assert_allclose(result.toarray(), expected, rtol=1e-10)

    def test_diagonal_matrix(self):
        """Test exponential of a diagonal matrix."""
        # Create a sparse diagonal matrix
        n = 3
        diag_values = np.array([1.0, 2.0, 3.0])
        diag_matrix = sparse.lil_matrix((n, n))
        for i in range(n):
            diag_matrix[i, i] = diag_values[i]

        # Compute the exponential
        result = compute_matrix_exponential(diag_matrix)

        # For a diagonal matrix, the exponential is a diagonal matrix with exponentials of the original values
        expected = sparse.lil_matrix((n, n))
        for i in range(n):
            expected[i, i] = np.exp(diag_values[i])

        np.testing.assert_allclose(result.toarray(), expected.toarray(), rtol=1e-10)

    def test_nilpotent_matrix(self):
        """Test exponential of a nilpotent matrix."""
        # Create a sparse nilpotent matrix (a matrix that becomes zero when raised to some power)
        n = 3
        nilpotent = sparse.lil_matrix((n, n))
        nilpotent[0, 1] = 1
        nilpotent[1, 2] = 1

        # Compute the exponential
        result = compute_matrix_exponential(nilpotent)

        # For this specific nilpotent matrix, we can compute the expected result
        # exp(N) = I + N + N²/2! + N³/3! + ...
        # Since N³ = 0 for this matrix, we have exp(N) = I + N + N²/2
        expected = sparse.eye(n) + nilpotent + (nilpotent @ nilpotent) / 2

        np.testing.assert_allclose(result.toarray(), expected.toarray(), rtol=1e-10)

    def test_general_matrix(self):
        """Test exponential of a general matrix against known result."""
        # Create a sparse matrix
        n = 2
        matrix = sparse.lil_matrix((n, n))
        matrix[0, 0] = 1
        matrix[0, 1] = 2
        matrix[1, 0] = 3
        matrix[1, 1] = 4

        # Compute the exponential
        result = compute_matrix_exponential(matrix)

        # Compute the expected result using numpy's matrix exponential for dense matrices
        dense_matrix = matrix.toarray()
        expected = linalg.expm(dense_matrix)

        np.testing.assert_allclose(result.toarray(), expected, rtol=1e-10)

    def test_return_type(self):
        """Test that the function returns a sparse matrix of the correct type."""
        n = 3
        matrix = sparse.lil_matrix((n, n))
        matrix[0, 1] = 1

        result = compute_matrix_exponential(matrix)

        # Check that the result is a sparse matrix
        self.assertTrue(sparse.issparse(result))


if __name__ == "__main__":
    unittest.main()
