import os
# Add the parent directory to the path so we can import the sample
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sympy
import sympy.physics.quantum
from sample_178 import custom_trace


class TestCustomTrace(unittest.TestCase):
    def test_custom_trace_returns_trace_object(self):
        """Test that custom_trace returns a Tr object."""
        result = custom_trace(5)
        self.assertIsInstance(result, sympy.physics.quantum.trace.Tr)
    
    def test_custom_trace_with_integer(self):
        """Test that custom_trace correctly handles integer input."""
        n = 10
        result = custom_trace(n)
        # Assuming custom_trace returns a Tr object with the integer as its argument
        self.assertIsInstance(result, sympy.physics.quantum.trace.Tr)
        self.assertEqual(result.args[0], n)
    
    def test_custom_trace_with_matrix(self):
        """Test that custom_trace works with a sympy matrix."""
        # Create a simple 2x2 matrix
        matrix = sympy.Matrix([[1, 2], [3, 4]])
        result = custom_trace(matrix)
        # Assuming custom_trace returns a Tr object with the matrix as its argument
        self.assertIsInstance(result, sympy.physics.quantum.trace.Tr)
        self.assertEqual(result.args[0], matrix)
    
    def test_custom_trace_with_symbol(self):
        """Test that custom_trace works with sympy symbols."""
        x = sympy.Symbol('x')
        result = custom_trace(x)
        # Assuming custom_trace returns a Tr object with the symbol as its argument
        self.assertIsInstance(result, sympy.physics.quantum.trace.Tr)
        self.assertEqual(result.args[0], x)


if __name__ == "__main__":
    unittest.main()
