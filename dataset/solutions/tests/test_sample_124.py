import os
# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.linalg import lu_factor, lu_solve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_124 import compute_lu_decomposition


class TestLUDecomposition(unittest.TestCase):
    
    def test_lu_decomposition_square_matrix(self):
        """Test LU decomposition on a square matrix."""
        # Create a test matrix
        A = np.array([[2, 5, 8, 7], 
                      [5, 2, 2, 8], 
                      [7, 5, 6, 6], 
                      [5, 4, 4, 8]])
        
        # Get the LU decomposition
        p, l, u = compute_lu_decomposition(A)
        
        # Verify dimensions
        self.assertEqual(p.shape, (4, 4))
        self.assertEqual(l.shape, (4, 4))
        self.assertEqual(u.shape, (4, 4))
        
        # Verify that P*A = L*U
        # Convert permutation matrix to actual permutation
        p_indices = np.argmax(p, axis=1)
        A_permuted = A[p_indices]
        
        # Check if L*U equals the permuted A
        LU = np.matmul(l, u)
        np.testing.assert_allclose(A_permuted, LU, rtol=1e-10, atol=1e-10)
        
        # Verify L is lower triangular with ones on the diagonal
        for i in range(l.shape[0]):
            for j in range(i+1, l.shape[1]):
                self.assertAlmostEqual(l[i, j], 0)
            self.assertAlmostEqual(l[i, i], 1)
        
        # Verify U is upper triangular
        for i in range(1, u.shape[0]):
            for j in range(i):
                self.assertAlmostEqual(u[i, j], 0)
    
    def test_lu_decomposition_rectangular_matrix(self):
        """Test LU decomposition on a rectangular matrix."""
        # Create a rectangular test matrix
        A = np.array([[2, 5, 8], 
                      [5, 2, 2], 
                      [7, 5, 6], 
                      [5, 4, 4]])
        
        # Get the LU decomposition
        p, l, u = compute_lu_decomposition(A)
        
        # Verify dimensions
        self.assertEqual(p.shape, (4, 4))
        self.assertEqual(l.shape, (4, 3))
        self.assertEqual(u.shape, (3, 3))
        
        # Verify that P*A = L*U
        # Convert permutation matrix to actual permutation
        p_indices = np.argmax(p, axis=1)
        A_permuted = A[p_indices]
        
        # Check if L*U equals the permuted A
        LU = np.matmul(l, u)
        np.testing.assert_allclose(A_permuted, LU, rtol=1e-10, atol=1e-10)
    
    def test_lu_decomposition_singular_matrix(self):
        """Test LU decomposition on a singular matrix."""
        # Create a singular matrix
        A = np.array([[1, 2, 3], 
                      [4, 5, 6], 
                      [7, 8, 9]])  # Third row is sum of first two
        
        # Get the LU decomposition
        p, l, u = compute_lu_decomposition(A)
        
        # Verify dimensions
        self.assertEqual(p.shape, (3, 3))
        self.assertEqual(l.shape, (3, 3))
        self.assertEqual(u.shape, (3, 3))
        
        # Verify that P*A = L*U
        # Convert permutation matrix to actual permutation
        p_indices = np.argmax(p, axis=1)
        A_permuted = A[p_indices]
        
        # Check if L*U equals the permuted A
        LU = np.matmul(l, u)
        np.testing.assert_allclose(A_permuted, LU, rtol=1e-10, atol=1e-10)
    
    def test_lu_decomposition_solve_system(self):
        """Test using LU decomposition to solve a linear system."""
        # Create a test matrix and vector
        A = np.array([[3, 1, 2], 
                      [6, 3, 4], 
                      [3, 1, 5]])
        b = np.array([5, 7, 8])
        
        # Get the LU decomposition
        p, l, u = compute_lu_decomposition(A)
        
        # Convert permutation matrix to permutation indices
        p_indices = np.argmax(p, axis=1)
        
        # Use scipy's lu_factor and lu_solve to get the expected solution
        lu_and_piv = lu_factor(A)
        expected_x = lu_solve(lu_and_piv, b)
        
        # Solve the system using our decomposition
        # First apply the permutation to b
        b_permuted = b[p_indices]
        
        # Solve L*y = P*b for y
        y = np.zeros_like(b, dtype=float)
        for i in range(len(b)):
            y[i] = b_permuted[i] - np.sum(l[i, :i] * y[:i])
        
        # Solve U*x = y for x
        x = np.zeros_like(b, dtype=float)
        for i in range(len(b)-1, -1, -1):
            x[i] = (y[i] - np.sum(u[i, i+1:] * x[i+1:])) / u[i, i]
        
        # Verify the solution
        np.testing.assert_allclose(x, expected_x, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.dot(A, x), b, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    unittest.main()