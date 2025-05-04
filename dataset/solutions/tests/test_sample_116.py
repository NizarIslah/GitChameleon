import os
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_116 import compute_matrix_exponential


class TestMatrixExponential(unittest.TestCase):
    def test_identity_matrix(self):
        """Test that exp(0) = I, where I is the identity matrix."""
        # Create a 3x3 zero matrix in LIL format
        n = 3
        A = sparse.lil_matrix((n, n))

        # Compute the matrix exponential
        result = compute_matrix_exponential(A)

        # The exponential of a zero matrix should be the identity matrix
        expected = sparse.eye(n).tolil()

        # Convert to arrays for comparison
        self.assertTrue(np.allclose(result.toarray(), expected.toarray()))

    def test_diagonal_matrix(self):
        """Test matrix exponential of a diagonal matrix."""
        # Create a diagonal matrix with [1, 2, 3] on the diagonal
        n = 3
        A = sparse.lil_matrix((n, n))
        A[0, 0] = 1
        A[1, 1] = 2
        A[2, 2] = 3

        # Compute the matrix exponential
        result = compute_matrix_exponential(A)

        # For a diagonal matrix, the exponential is a diagonal matrix with
        # the exponential of each diagonal element
        expected = sparse.lil_matrix((n, n))
        expected[0, 0] = np.exp(1)
        expected[1, 1] = np.exp(2)
        expected[2, 2] = np.exp(3)

        # Compare results
        self.assertTrue(np.allclose(result.toarray(), expected.toarray()))

    def test_nilpotent_matrix(self):
        """Test matrix exponential of a nilpotent matrix."""
        # Create a nilpotent matrix (a matrix that becomes zero when raised to some power)
        n = 3
        A = sparse.lil_matrix((n, n))
        A[0, 1] = 1
        A[1, 2] = 1
        # A^3 = 0 for this matrix

        # Compute the matrix exponential
        result = compute_matrix_exponential(A)

        # For this nilpotent matrix, exp(A) = I + A + A^2/2
        expected = sparse.lil_matrix((n, n))
        expected[0, 0] = 1
        expected[1, 1] = 1
        expected[2, 2] = 1
        expected[0, 1] = 1
        expected[1, 2] = 1
        expected[0, 2] = 0.5  # From A^2/2

        # Compare results
        self.assertTrue(np.allclose(result.toarray(), expected.toarray()))


if __name__ == "__main__":
    unittest.main()
