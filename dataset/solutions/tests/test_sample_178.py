import unittest
import sys
import os

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sympy
import sympy.physics.quantum
from sympy.physics.quantum.trace import Tr
import sample_178


class TestCustomTrace(unittest.TestCase):
    """Test cases for the custom_trace function in sample_178.py."""

    def test_return_value(self):
        """Test that custom_trace returns the expected value."""
        # In the current version of SymPy, Tr(n) returns n directly
        result = sample_178.custom_trace(5)
        self.assertEqual(result, 5)

    def test_with_integer_input(self):
        """Test the function with various integer inputs."""
        test_values = [1, 2, 5, 10, 100]
        
        for n in test_values:
            result = sample_178.custom_trace(n)
            
            # Check that the result equals the input
            self.assertEqual(result, n)

    def test_with_zero(self):
        """Test the function with zero as input."""
        result = sample_178.custom_trace(0)
        
        # Check that the result equals 0
        self.assertEqual(result, 0)

    def test_with_negative_integer(self):
        """Test the function with negative integer input."""
        result = sample_178.custom_trace(-5)
        
        # Check that the result equals -5
        self.assertEqual(result, -5)

    def test_matches_direct_tr_creation(self):
        """Test that our function matches direct creation of Tr objects."""
        test_values = [1, 2, 5, 10, 100]
        
        for n in test_values:
            # Our implementation
            our_result = sample_178.custom_trace(n)
            
            # Direct creation
            direct_result = Tr(n)
            
            # Check that the results are equal
            self.assertEqual(our_result, direct_result)

    def test_trace_with_arithmetic(self):
        """Test that the trace function works with arithmetic operations."""
        # Test addition
        self.assertEqual(sample_178.custom_trace(3) + sample_178.custom_trace(5), 8)
        
        # Test subtraction
        self.assertEqual(sample_178.custom_trace(10) - sample_178.custom_trace(3), 7)
        
        # Test multiplication
        self.assertEqual(sample_178.custom_trace(4) * sample_178.custom_trace(5), 20)
        
        # Test division
        self.assertEqual(sample_178.custom_trace(10) / sample_178.custom_trace(2), 5)

    def test_trace_with_sympy_expressions(self):
        """Test the function with SymPy expressions."""
        # Create a SymPy symbol
        x = sympy.symbols('x')
        
        # Create a SymPy expression
        expr = x**2 + 2*x + 1
        
        # Call the function with the expression
        result = sample_178.custom_trace(expr)
        
        # Check that the result equals the expression
        self.assertEqual(result, expr)


if __name__ == '__main__':
    unittest.main()