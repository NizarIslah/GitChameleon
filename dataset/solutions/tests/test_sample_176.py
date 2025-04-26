import unittest
import sys
import os

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sympy
from sympy.matrices.expressions.fourier import DFT
import sample_176


class TestCustomComputeDFT(unittest.TestCase):
    """Test cases for the custom_computeDFT function in sample_176.py."""

    def test_return_type(self):
        """Test that custom_computeDFT returns a SymPy ImmutableDenseMatrix."""
        result = sample_176.custom_computeDFT(2)
        self.assertIsInstance(result, sympy.ImmutableDenseMatrix)

    def test_matrix_size(self):
        """Test that the returned matrix has the correct dimensions."""
        test_sizes = [1, 2, 4, 8]
        
        for n in test_sizes:
            result = sample_176.custom_computeDFT(n)
            self.assertEqual(result.shape, (n, n))

    def test_dft_properties(self):
        """Test that the DFT matrix has expected mathematical properties."""
        # Test for n=4
        n = 4
        dft_matrix = sample_176.custom_computeDFT(n)
        
        # DFT matrix should be unitary (U* × U = I, where U* is the conjugate transpose)
        # For a unitary matrix, U × U.H = I where U.H is the Hermitian conjugate
        identity = dft_matrix * dft_matrix.H
        
        # Convert to numerical values for easier comparison
        numerical_identity = identity.evalf()
        expected_identity = sympy.eye(n)
        
        # Check if close to identity matrix (allowing for numerical precision issues)
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.assertTrue(abs(numerical_identity[i, j] - 1) < 1e-10)
                else:
                    self.assertTrue(abs(numerical_identity[i, j]) < 1e-10)

    def test_dft_values_small_case(self):
        """Test the actual values of the DFT matrix for a small case."""
        # Test for n=2
        result = sample_176.custom_computeDFT(2)
        
        # Expected DFT matrix for n=2
        # [1, 1]
        # [1, -1]
        # Normalized by 1/sqrt(2)
        expected = sympy.Matrix([[1, 1], [1, -1]]) / sympy.sqrt(2)
        
        # Convert to ImmutableDenseMatrix for comparison
        expected = sympy.ImmutableDenseMatrix(expected)
        
        # Check if matrices are equal
        self.assertEqual(result, expected)

    def test_dft_idempotent_property(self):
        """Test that applying DFT four times returns the original matrix (up to scaling)."""
        # DFT^4 = I (identity matrix)
        n = 4
        dft = sample_176.custom_computeDFT(n)
        
        # Apply DFT four times
        result = dft * dft * dft * dft
        
        # Expected result is identity matrix
        expected = sympy.eye(n)
        expected = sympy.ImmutableDenseMatrix(expected)
        
        # Check if matrices are equal
        self.assertEqual(result, expected)

    def test_matches_sympy_dft(self):
        """Test that our function matches SymPy's DFT implementation."""
        test_sizes = [1, 2, 4, 8]
        
        for n in test_sizes:
            # Our implementation
            our_result = sample_176.custom_computeDFT(n)
            
            # Direct SymPy implementation
            sympy_result = DFT(n).as_explicit()
            
            # Check if matrices are equal
            self.assertEqual(our_result, sympy_result)

    def test_handles_edge_case_n_equals_one(self):
        """Test that the function correctly handles n=1."""
        result = sample_176.custom_computeDFT(1)
        
        # DFT for n=1 should be [1]
        expected = sympy.ImmutableDenseMatrix([[1]])
        
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()