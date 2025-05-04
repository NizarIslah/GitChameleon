import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_193 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_193 import custom_function
from sympy import Matrix, symbols


class TestCustomFunction(unittest.TestCase):
    def test_flatten_2x2_matrix(self):
        """Test flattening a 2x2 matrix."""
        # Create a 2x2 matrix
        matrix = Matrix([[1, 2], [3, 4]])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = [1, 2, 3, 4]
        self.assertEqual(result, expected)

    def test_flatten_row_matrix(self):
        """Test flattening a row matrix."""
        # Create a row matrix
        matrix = Matrix([[1, 2, 3]])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = [1, 2, 3]
        self.assertEqual(result, expected)

    def test_flatten_column_matrix(self):
        """Test flattening a column matrix."""
        # Create a column matrix
        matrix = Matrix([[1], [2], [3]])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = [1, 2, 3]
        self.assertEqual(result, expected)

    def test_flatten_matrix_with_symbolic_elements(self):
        """Test flattening a matrix with symbolic elements."""
        # Create symbolic variables
        x, y, z, w = symbols("x y z w")

        # Create a matrix with symbolic elements
        matrix = Matrix([[x, y], [z, w]])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = [x, y, z, w]
        self.assertEqual(result, expected)

    def test_return_type_is_list(self):
        """Test that the return type is a list."""
        # Create a simple matrix
        matrix = Matrix([[1, 2], [3, 4]])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check that the result is a list
        self.assertIsInstance(result, list)

    def test_flatten_empty_matrix(self):
        """Test flattening an empty matrix."""
        # Create an empty matrix
        matrix = Matrix([])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = []
        self.assertEqual(result, expected)

    def test_flatten_large_matrix(self):
        """Test flattening a large matrix."""
        # Create a 5x5 matrix
        matrix = Matrix([[i + j * 5 for i in range(1, 6)] for j in range(5)])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = list(range(1, 26))
        self.assertEqual(result, expected)

    def test_flatten_matrix_with_mixed_types(self):
        """Test flattening a matrix with mixed types."""
        # Create symbolic variables
        x, y = symbols("x y")

        # Create a matrix with mixed types
        matrix = Matrix([[1, x], [y, 4]])

        # Flatten the matrix
        result = custom_function(matrix)

        # Check the result
        expected = [1, x, y, 4]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
