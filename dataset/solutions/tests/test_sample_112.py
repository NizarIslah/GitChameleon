import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_112 import compute_matrix_exponential


class TestMatrixExponential(unittest.TestCase):
    def test_identity_matrix(self):
        """Test that exp(0) = I for matrix exponential"""
        # Create a zero matrix
        zero_matrix = np.zeros((3, 3))
        result = compute_matrix_exponential(zero_matrix)
        # The exponential of a zero matrix should be the identity matrix
        expected = np.eye(3)
        np.testing.assert_allclose(result, expected)

    def test_diagonal_matrix(self):
        """Test matrix exponential of a diagonal matrix"""
        # Create a diagonal matrix
        diag_matrix = np.diag([1.0, 2.0, 3.0])
        result = compute_matrix_exponential(diag_matrix)
        # For diagonal matrices, the exponential is just the exponential of each element
        expected = np.diag([np.exp(1.0), np.exp(2.0), np.exp(3.0)])
        np.testing.assert_allclose(result, expected)

    def test_against_scipy_implementation(self):
        """Test our function against SciPy's implementation directly"""
        # Create a random matrix
        np.random.seed(42)  # For reproducibility
        random_matrix = np.random.rand(4, 4)

        # Compare our function with SciPy's implementation
        result = compute_matrix_exponential(random_matrix)
        expected = linalg.expm(random_matrix)
        np.testing.assert_allclose(result, expected)

    def test_known_result(self):
        """Test with a matrix that has a known exponential result"""
        # Simple 2x2 matrix with known result
        matrix = np.array([[0, 1], [-1, 0]])  # This represents a 90-degree rotation
        result = compute_matrix_exponential(matrix)

        # The exponential of this matrix should approximate cos(1) and sin(1) values
        expected = np.array([[np.cos(1), np.sin(1)], [-np.sin(1), np.cos(1)]])
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_input_validation(self):
        """Test that the function handles different input types correctly"""
        # Test with a list that can be converted to ndarray
        matrix_list = [[1, 0], [0, 1]]
        result = compute_matrix_exponential(np.array(matrix_list))
        expected = np.array([[np.exp(1), 0], [0, np.exp(1)]])
        np.testing.assert_allclose(result, expected)

        # Test with a 1D array should raise an error in SciPy's expm
        with self.assertRaises(Exception):
            compute_matrix_exponential(np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
