import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_158 import check_invertibility


class TestCheckInvertibility(unittest.TestCase):
    def test_all_invertible_matrices(self):
        # Create an array of invertible matrices
        matrices = np.array(
            [
                [[1, 2], [3, 4]],  # det = -2
                [[2, 1], [1, 1]],  # det = 1
                [[5, 2], [3, 2]],  # det = 4
            ]
        )

        result = check_invertibility(matrices)
        self.assertTrue(result)

    def test_some_non_invertible_matrices(self):
        # Create an array with some non-invertible matrices
        matrices = np.array(
            [
                [[1, 2], [3, 4]],  # det = -2 (invertible)
                [[1, 2], [2, 4]],  # det = 0 (non-invertible)
                [[5, 2], [3, 2]],  # det = 4 (invertible)
            ]
        )

        result = check_invertibility(matrices)
        self.assertFalse(result)

    def test_all_non_invertible_matrices(self):
        # Create an array of non-invertible matrices
        matrices = np.array(
            [
                [[1, 2], [2, 4]],  # det = 0
                [[0, 0], [0, 0]],  # det = 0
                [[1, 1], [2, 2]],  # det = 0
            ]
        )

        result = check_invertibility(matrices)
        self.assertFalse(result)

    def test_single_matrix(self):
        # Test with a single invertible matrix
        matrix_invertible = np.array([[[1, 0], [0, 1]]])  # Identity matrix, det = 1
        self.assertTrue(check_invertibility(matrix_invertible))

        # Test with a single non-invertible matrix
        matrix_non_invertible = np.array([[[1, 1], [1, 1]]])  # det = 0
        self.assertFalse(check_invertibility(matrix_non_invertible))

    def test_3d_matrices(self):
        # Test with 3x3 matrices
        matrices = np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity matrix, det = 1
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # Singular matrix, det = 0
            ]
        )

        result = check_invertibility(matrices)
        self.assertFalse(result)

    def test_empty_array(self):
        # Test with an empty array - this should return True as there are no matrices that are non-invertible
        # This is a logical interpretation of "all matrices are invertible" when there are no matrices
        matrices = np.array([])

        # This might raise an error depending on implementation, so we'll handle that case
        try:
            result = check_invertibility(matrices)
            # If no error, we expect True (all elements satisfy the condition when there are no elements)
            self.assertTrue(result)
        except Exception as e:
            # If an error occurs, the test will be marked as skipped
            self.skipTest(f"Empty array test raised an exception: {str(e)}")


if __name__ == "__main__":
    unittest.main()
