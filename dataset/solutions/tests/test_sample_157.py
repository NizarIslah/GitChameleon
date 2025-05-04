import os

# Add the parent directory to the path so we can import the module
import sys
import unittest
import warnings

import numpy as np
from scipy.linalg import det

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_157 import check_invertibility


class TestCheckInvertibility(unittest.TestCase):
    def setUp(self):
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")

    def test_single_invertible_matrix(self):
        # Test with a single invertible matrix
        matrix = np.array([[1, 2], [3, 4]])
        result = check_invertibility(matrix)
        self.assertTrue(result)

    def test_single_non_invertible_matrix(self):
        # Test with a single non-invertible (singular) matrix
        matrix = np.array([[1, 2], [2, 4]])  # Determinant is zero
        result = check_invertibility(matrix)
        self.assertFalse(result)

    def test_multiple_matrices_all_invertible(self):
        # Test with multiple invertible matrices
        matrices = np.array(
            [
                [[1, 2], [3, 4]],  # Determinant is -2
                [[2, 1], [1, 3]],  # Determinant is 5
            ]
        )
        result = check_invertibility(matrices)
        self.assertTrue(result)

    def test_multiple_matrices_one_non_invertible(self):
        # Test with multiple matrices where one is non-invertible
        matrices = np.array(
            [[[1, 2], [3, 4]], [[1, 2], [2, 4]]]  # Invertible  # Non-invertible
        )
        result = check_invertibility(matrices)
        self.assertFalse(result)

    def test_3d_matrices(self):
        # Test with 3x3 matrices
        matrices = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 10]],  # Invertible
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity matrix (invertible)
            ]
        )
        result = check_invertibility(matrices)
        self.assertTrue(result)

    def test_mixed_invertibility_3d_matrices(self):
        # Test with 3x3 matrices where one is non-invertible
        matrices = np.array(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],  # Non-invertible (linearly dependent rows)
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity matrix (invertible)
            ]
        )
        result = check_invertibility(matrices)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
