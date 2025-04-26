import unittest
import sys
import os

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sympy
from sympy import symbols, eye, Matrix, ImmutableDenseMatrix
import sample_177


class TestCustomLaplaceTransform(unittest.TestCase):
    """Test cases for the custom_laplace_transform function in sample_177.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Create symbols for testing
        self.t = symbols('t', real=True, positive=True)
        self.z = symbols('z')

    def test_return_type(self):
        """Test that custom_laplace_transform returns a tuple with the correct types."""
        result = sample_177.custom_laplace_transform(self.t, self.z)
        
        # Check that the result is a tuple
        self.assertIsInstance(result, tuple)
        
        # Check that the tuple has 3 elements
        self.assertEqual(len(result), 3)
        
        # Check the types of the tuple elements
        self.assertIsInstance(result[0], Matrix)
        # The second element might be a sympy.Expr or a sympy.core.relational.Relational
        self.assertTrue(isinstance(result[1], sympy.Expr) or isinstance(result[1], sympy.core.relational.Relational))
        # The third element is True, which is a bool value
        self.assertEqual(result[2], True)

    def test_matrix_size(self):
        """Test that the returned matrix has the correct dimensions."""
        result = sample_177.custom_laplace_transform(self.t, self.z)
        
        # Check that the matrix is 2x2
        self.assertEqual(result[0].shape, (2, 2))

    def test_matrix_values(self):
        """Test that the matrix values are correct."""
        result = sample_177.custom_laplace_transform(self.t, self.z)
        
        # The Laplace transform of the identity matrix should be 1/z times the identity matrix
        expected_matrix = eye(2) / self.z
        
        # Check that the matrices are equal
        self.assertEqual(result[0], expected_matrix)

    def test_convergence_region(self):
        """Test that the convergence region is correct."""
        result = sample_177.custom_laplace_transform(self.t, self.z)
        
        # Get the actual convergence region from the result
        actual_region = result[1]
        
        # For the Laplace transform of the identity matrix, the convergence region
        # might be represented differently depending on the SymPy version
        # We'll check if it's 0 (which means no constraints) or Re(z) > 0
        self.assertTrue(actual_region == 0 or actual_region == sympy.re(self.z) > 0,
                        f"Expected 0 or Re(z) > 0, got {actual_region}")

    def test_boolean_value(self):
        """Test that the boolean value is correct."""
        result = sample_177.custom_laplace_transform(self.t, self.z)
        
        # The boolean value should be True
        self.assertTrue(result[2])

    def test_with_different_symbols(self):
        """Test the function with different symbols."""
        # Create different symbols
        s = symbols('s')
        tau = symbols('tau', real=True, positive=True)
        
        # Call the function with different symbols
        result = sample_177.custom_laplace_transform(tau, s)
        
        # Check that the result is correct
        expected_matrix = eye(2) / s
        
        # Check the matrix
        self.assertEqual(result[0], expected_matrix)
        
        # Check the convergence region (might be 0 or Re(s) > 0)
        actual_region = result[1]
        self.assertTrue(actual_region == 0 or actual_region == sympy.re(s) > 0,
                        f"Expected 0 or Re(s) > 0, got {actual_region}")
        
        # Check the boolean value
        self.assertEqual(result[2], True)

    def test_matches_direct_laplace_transform(self):
        """Test that our function matches direct use of laplace_transform."""
        # Our function result
        our_result = sample_177.custom_laplace_transform(self.t, self.z)
        
        # Direct use of laplace_transform
        direct_result = sympy.laplace_transform(eye(2), self.t, self.z, legacy_matrix=False)
        
        # Check that the results are equal
        self.assertEqual(our_result, direct_result)


if __name__ == '__main__':
    unittest.main()