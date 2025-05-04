import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sympy
from sample_178 import custom_trace


class TestCustomTrace(unittest.TestCase):
    def test_custom_trace_returns_input_for_integer(self):
        """Test that custom_trace returns the same integer that was passed in."""
        result = custom_trace(5)
        self.assertEqual(result, 5)

    def test_custom_trace_with_integer(self):
        """Test that custom_trace returns the same integer that was passed in."""
        n = 10
        result = custom_trace(n)
        self.assertEqual(result, n)

    def test_custom_trace_with_matrix(self):
        """Test that custom_trace returns 5 for a sympy Matrix."""
        matrix = sympy.Matrix([[1, 2], [3, 4]])
        result = custom_trace(matrix)
        self.assertEqual(result, 5)

    def test_custom_trace_with_symbol(self):
        """Test that custom_trace returns the same symbol that was passed in."""
        x = sympy.Symbol("x")
        result = custom_trace(x)
        self.assertEqual(result, x)


if __name__ == "__main__":
    unittest.main()
