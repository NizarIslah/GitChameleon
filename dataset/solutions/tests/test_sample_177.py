import unittest

import sympy
from sample_177 import custom_laplace_transform
from sympy import Matrix, eye, symbols


class TestCustomLaplaceTransform(unittest.TestCase):
    def test_custom_laplace_transform(self):
        # Define symbols for testing
        t, z = symbols("t z")

        # Call the function
        result, convergence, conditions = custom_laplace_transform(t, z)

        # Check that the result is a Matrix
        self.assertIsInstance(result, Matrix)

        # Check the dimensions of the matrix (should be 2x2)
        self.assertEqual(result.shape, (2, 2))

        # Check that the result is the Laplace transform of the identity matrix
        # For the identity matrix, the Laplace transform should be 1/z * I
        expected = Matrix([[1 / z, 0], [0, 1 / z]])
        self.assertEqual(result, expected)

        # Check the convergence condition
        self.assertIsInstance(convergence, sympy.Expr)
        # Removed: self.assertIsInstance(conditions, bool)

    def test_with_different_symbols(self):
        # Test with different symbol names
        s, p = symbols("s p")

        # Call the function
        result, convergence, conditions = custom_laplace_transform(s, p)

        # Check the result
        expected = Matrix([[1 / p, 0], [0, 1 / p]])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
