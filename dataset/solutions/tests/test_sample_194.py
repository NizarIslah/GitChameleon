import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_194 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_194 import custom_function
from sympy import Matrix, symbols


class TestCustomFunction(unittest.TestCase):
    def test_convert_2x2_matrix_to_dictionary(self):
        """Test converting a 2x2 matrix to a dictionary."""
        # Create a 2x2 matrix
        matrix = Matrix([[1, 2], [3, 4]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check the result
        expected = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        self.assertEqual(result, expected)

    def test_convert_row_matrix_to_dictionary(self):
        """Test converting a row matrix to a dictionary."""
        # Create a row matrix
        matrix = Matrix([[1, 2, 3]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check the result
        expected = {(0, 0): 1, (0, 1): 2, (0, 2): 3}
        self.assertEqual(result, expected)

    def test_convert_column_matrix_to_dictionary(self):
        """Test converting a column matrix to a dictionary."""
        # Create a column matrix
        matrix = Matrix([[1], [2], [3]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check the result
        expected = {(0, 0): 1, (1, 0): 2, (2, 0): 3}
        self.assertEqual(result, expected)

    def test_convert_matrix_with_symbolic_elements(self):
        """Test converting a matrix with symbolic elements to a dictionary."""
        # Create symbolic variables
        x, y, z, w = symbols("x y z w")

        # Create a matrix with symbolic elements
        matrix = Matrix([[x, y], [z, w]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check the result
        expected = {(0, 0): x, (0, 1): y, (1, 0): z, (1, 1): w}
        self.assertEqual(result, expected)

    def test_return_type_is_dictionary(self):
        """Test that the return type is a dictionary."""
        # Create a simple matrix
        matrix = Matrix([[1, 2], [3, 4]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)

    def test_convert_empty_matrix_to_dictionary(self):
        """Test converting an empty matrix to a dictionary."""
        # Create an empty matrix
        matrix = Matrix([])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check the result
        expected = {}
        self.assertEqual(result, expected)

    def test_dictionary_keys_are_tuples_of_indices(self):
        """Test that the dictionary keys are tuples of indices."""
        # Create a matrix
        matrix = Matrix([[1, 2], [3, 4]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check that all keys are tuples of indices
        for key in result.keys():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            self.assertIsInstance(key[0], int)
            self.assertIsInstance(key[1], int)

    def test_zero_elements_are_not_included(self):
        """Test that zero elements are not included in the dictionary."""
        # Create a matrix with zero elements
        matrix = Matrix([[1, 0], [0, 4]])

        # Convert the matrix to a dictionary
        result = custom_function(matrix)

        # Check the result
        expected = {(0, 0): 1, (1, 1): 4}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
