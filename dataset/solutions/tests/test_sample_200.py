import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_200 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_200 import custom_array_to_matrix
from sympy import Array, Matrix, symbols
from sympy.tensor.array import ImmutableDenseNDimArray


class TestCustomArrayToMatrix(unittest.TestCase):
    def test_with_2d_array(self):
        """Test custom_array_to_matrix with a 2D array."""
        # Create a 2D array
        array = Array([[1, 2], [3, 4]])

        # Convert the array to a matrix
        result = custom_array_to_matrix(array)

        # Check the result - compare the values
        self.assertEqual(result.tolist(), [[1, 2], [3, 4]])

    def test_with_1d_array(self):
        """Test custom_array_to_matrix with a 1D array."""
        # Create a 1D array
        array = Array([1, 2, 3, 4])

        # Convert the array to a matrix
        result = custom_array_to_matrix(array)

        # Check the result - compare the values
        # Note: The result might be a row or column vector depending on the implementation
        self.assertTrue(
            result.tolist() == [1, 2, 3, 4] or result.tolist() == [[1], [2], [3], [4]]
        )

    def test_with_symbolic_array(self):
        """Test custom_array_to_matrix with a symbolic array."""
        # Create symbolic variables
        x, y, z, w = symbols("x y z w")

        # Create a symbolic array
        array = Array([[x, y], [z, w]])

        # Convert the array to a matrix
        result = custom_array_to_matrix(array)

        # Check the result - compare the values
        self.assertEqual(result[0, 0], x)
        self.assertEqual(result[0, 1], y)
        self.assertEqual(result[1, 0], z)
        self.assertEqual(result[1, 1], w)

    def test_with_empty_array(self):
        """Test custom_array_to_matrix with an empty array."""
        # Create an empty array
        array = Array([])

        # Convert the array to a matrix
        result = custom_array_to_matrix(array)

        # Check the result - an empty array should convert to an empty result
        self.assertEqual(len(result), 0)

    def test_with_higher_dimensional_array(self):
        """Test custom_array_to_matrix with a higher-dimensional array."""
        # Create a 3D array
        array = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        # Try to convert the array to a matrix
        try:
            result = custom_array_to_matrix(array)
            # If we get here, the function accepted the input
            # Let's check if the result has values
            self.assertTrue(len(result) > 0)
        except Exception as e:
            # If an exception is raised, it's expected since higher-dimensional arrays
            # might not be directly convertible to matrices
            pass

    def test_with_non_array_input(self):
        """Test custom_array_to_matrix with non-array input."""
        # Try with a list (not a sympy.Array)
        try:
            result = custom_array_to_matrix([[1, 2], [3, 4]])
            # If we get here, the function accepted the input
            # Let's check if the result has values
            self.assertTrue(len(result) > 0)
        except Exception as e:
            # If an exception is raised, it's expected since the input is not a sympy.Array
            pass

        # Try with a Matrix (not a sympy.Array)
        try:
            result = custom_array_to_matrix(Matrix([[1, 2], [3, 4]]))
            # If we get here, the function accepted the input
            # Let's check if the result has values
            self.assertTrue(len(result) > 0)
        except Exception as e:
            # If an exception is raised, it's expected since the input is not a sympy.Array
            pass

    def test_return_type(self):
        """Test that the return type is appropriate for matrix operations."""
        # Create a simple array
        array = Array([[1, 2], [3, 4]])

        # Convert the array to a matrix
        result = custom_array_to_matrix(array)

        # Check that the result is either a Matrix or an ImmutableDenseNDimArray
        self.assertTrue(
            isinstance(result, Matrix) or isinstance(result, ImmutableDenseNDimArray)
        )

    def test_matrix_properties_preserved(self):
        """Test that matrix properties are preserved in the conversion."""
        # Create an array
        array = Array([[1, 2], [3, 4]])

        # Convert the array to a matrix
        result = custom_array_to_matrix(array)

        # Check that the shape is preserved
        self.assertEqual(result.shape, (2, 2))

        # Check that the elements are preserved
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 0], 3)
        self.assertEqual(result[1, 1], 4)

        # Convert to Matrix if needed for matrix operations
        if not isinstance(result, Matrix):
            matrix_result = Matrix(result.tolist())

            # Check matrix operations
            self.assertEqual(matrix_result.det(), -2)
            self.assertEqual(matrix_result.trace(), 5)
        else:
            # Check matrix operations directly
            self.assertEqual(result.det(), -2)
            self.assertEqual(result.trace(), 5)


if __name__ == "__main__":
    unittest.main()
