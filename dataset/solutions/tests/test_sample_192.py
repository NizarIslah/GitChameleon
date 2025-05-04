import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_192 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_192 import custom_create_matrix
from sympy import Matrix, symbols


class TestCustomCreateMatrix(unittest.TestCase):
    def test_create_matrix_from_two_row_matrices(self):
        """Test creating a matrix from two row matrices."""
        # Create two row matrices
        first = Matrix([[1, 2, 3]])
        second = Matrix([[4, 5, 6]])

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check the result
        expected = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(result, expected)

    def test_create_matrix_from_two_column_matrices(self):
        """Test creating a matrix from two column matrices."""
        # Create two column matrices
        first = Matrix([[1], [2], [3]])
        second = Matrix([[4], [5], [6]])

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check the result
        expected = Matrix([[1], [2], [3], [4], [5], [6]])
        self.assertEqual(result, expected)

    def test_create_matrix_from_matrices_with_different_dimensions(self):
        """Test creating a matrix from matrices with different dimensions."""
        # Create matrices with different dimensions but same number of columns
        first = Matrix([[1, 2], [3, 4]])  # 2x2 matrix
        second = Matrix([[5, 6], [7, 8], [9, 10]])  # 3x2 matrix

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check the result
        expected = Matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        self.assertEqual(result, expected)

    def test_return_type_is_matrix(self):
        """Test that the return type is a Matrix."""
        # Create two simple matrices
        first = Matrix([[1, 2]])
        second = Matrix([[3, 4]])

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check that the result is a Matrix
        self.assertIsInstance(result, Matrix)

    def test_matrix_dimensions_are_correct(self):
        """Test that the dimensions of the resulting matrix are correct."""
        # Create two matrices
        first = Matrix([[1, 2], [3, 4]])  # 2x2 matrix
        second = Matrix([[5, 6], [7, 8]])  # 2x2 matrix

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check the dimensions
        self.assertEqual(result.shape, (4, 2))  # Should be a 4x2 matrix

    def test_matrix_elements_are_preserved(self):
        """Test that the elements of the original matrices are preserved in the result."""
        # Create two matrices with specific elements
        first = Matrix([[1, 2], [3, 4]])
        second = Matrix([[5, 6], [7, 8]])

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check that all elements from the original matrices are in the result
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 0], 3)
        self.assertEqual(result[1, 1], 4)
        self.assertEqual(result[2, 0], 5)
        self.assertEqual(result[2, 1], 6)
        self.assertEqual(result[3, 0], 7)
        self.assertEqual(result[3, 1], 8)

    def test_create_matrix_with_symbolic_elements(self):
        """Test creating a matrix with symbolic elements."""
        # Create symbolic variables
        x, y, z, w = symbols("x y z w")

        # Create matrices with symbolic elements
        first = Matrix([[x, y]])
        second = Matrix([[z, w]])

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check the result
        expected = Matrix([[x, y], [z, w]])
        self.assertEqual(result, expected)

    def test_create_matrix_with_empty_matrices(self):
        """Test creating a matrix with empty matrices."""
        # Create empty matrices
        first = Matrix([])
        second = Matrix([])

        # Create the combined matrix
        result = custom_create_matrix(first, second)

        # Check the result
        expected = Matrix([])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
