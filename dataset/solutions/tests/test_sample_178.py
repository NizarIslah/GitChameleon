import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sympy
from sample_178 import custom_trace


class TestCustomTrace(unittest.TestCase):
    def test_custom_trace_with_integer(self):
        """Test that custom_trace returns the integer input unchanged."""
        n = 10
        result = custom_trace(n)
        self.assertEqual(result, n)
    
    def test_custom_trace_with_matrix(self):
        """Test that custom_trace returns the matrix input unchanged."""
        matrix = sympy.Matrix([[1, 2], [3, 4]])
        result = custom_trace(matrix)
        self.assertEqual(result, matrix)
    
    def test_custom_trace_with_symbol(self):
        """Test that custom_trace returns the symbol input unchanged."""
        x = sympy.Symbol('x')
        result = custom_trace(x)
        self.assertEqual(result, x)


if __name__ == "__main__":
    unittest.main()
