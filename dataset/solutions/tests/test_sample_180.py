# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_180 import custom_parse_mathematica
from sympy import Integer, Max, Min, symbols


class TestCustomParseMathematica(unittest.TestCase):
    def test_basic_parsing(self):
        """Test basic Mathematica expression parsing without F function."""
        result = custom_parse_mathematica("2 + 3")
        self.assertEqual(result, Integer(5))

        result = custom_parse_mathematica("x^2 + y^2")
        x, y = symbols("x y")
        self.assertEqual(result, x**2 + y**2)

    def test_f_function_replacement(self):
        """Test replacement of F function with Max(*x)*Min(*x)."""
        # Test F[a, b] which should become Max(a, b) * Min(a, b)
        result = custom_parse_mathematica("F[3, 5]")
        self.assertEqual(result, Integer(15))  # Max(3,5)*Min(3,5) = 5*3 = 15

        # Test F[a, b, c] which should become Max(a,b,c) * Min(a,b,c)
        result = custom_parse_mathematica("F[2, 4, 6]")
        self.assertEqual(result, Integer(12))  # Max(2,4,6)*Min(2,4,6) = 6*2 = 12

        # Test F function with symbolic arguments
        result = custom_parse_mathematica("F[x, y]")
        x, y = symbols("x y")
        expected = Max(x, y) * Min(x, y)
        self.assertEqual(result, expected)

    def test_complex_expressions(self):
        """Test more complex expressions involving the F function."""
        # Test F function within a larger expression
        result = custom_parse_mathematica("2 * F[3, 5] + 1")
        self.assertEqual(result, Integer(31))  # 2 * (5*3) + 1 = 2*15 + 1 = 31

        # Test nested F functions
        result = custom_parse_mathematica("F[F[1, 2], 3]")
        # F[1, 2] = Max(1,2)*Min(1,2) = 2*1 = 2
        # F[2, 3] = Max(2,3)*Min(2,3) = 3*2 = 6
        self.assertEqual(result, Integer(6))

        # Test with arithmetic operations inside F
        result = custom_parse_mathematica("F[1+2, 3*2]")
        self.assertEqual(result, Integer(18))  # F[3, 6] = Max(3,6)*Min(3,6) = 6*3 = 18


if __name__ == "__main__":
    unittest.main()
