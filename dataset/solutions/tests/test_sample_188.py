import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_188 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_188 import custom_generatePolyList
from sympy import symbols, Poly


class TestCustomGeneratePolyList(unittest.TestCase):
    def test_basic_polynomial(self):
        """Test basic polynomial conversion to list."""
        # Create a symbol
        x = symbols("x")

        # Create a simple polynomial: x^2 + 2x + 3
        poly = Poly(x**2 + 2 * x + 3, x)

        # Test the function
        # Expected result: [1, 2, 3] (coefficients in descending order of power)
        result = custom_generatePolyList(poly)

        # Check the result
        self.assertEqual(result, [1, 2, 3])

    def test_higher_degree_polynomial(self):
        """Test polynomial with higher degree."""
        # Create a symbol
        x = symbols("x")

        # Create a higher degree polynomial: x^4 + 2x^3 + 3x^2 + 4x + 5
        poly = Poly(x**4 + 2 * x**3 + 3 * x**2 + 4 * x + 5, x)

        # Test the function
        # Expected result: [1, 2, 3, 4, 5] (coefficients in descending order of power)
        result = custom_generatePolyList(poly)

        # Check the result
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_polynomial_with_zero_coefficients(self):
        """Test polynomial with zero coefficients."""
        # Create a symbol
        x = symbols("x")

        # Create a polynomial with zero coefficients: x^3 + 0*x^2 + 2*x + 0
        poly = Poly(x**3 + 0 * x**2 + 2 * x + 0, x)

        # Test the function
        # Expected result: [1, 0, 2, 0] (coefficients in descending order of power)
        result = custom_generatePolyList(poly)

        # Check the result
        self.assertEqual(result, [1, 0, 2, 0])

    def test_constant_polynomial(self):
        """Test constant polynomial."""
        # Create a symbol
        x = symbols("x")

        # Create a constant polynomial: 5
        poly = Poly(5, x)

        # Test the function
        # Expected result: [5] (just the constant term)
        result = custom_generatePolyList(poly)

        # Check the result
        self.assertEqual(result, [5])

    def test_zero_polynomial(self):
        """Test zero polynomial."""
        # Create a symbol
        x = symbols("x")

        # Create a zero polynomial: 0
        poly = Poly(0, x)

        # Test the function
        # For a zero polynomial, SymPy returns an empty list
        result = custom_generatePolyList(poly)

        # Check the result
        self.assertEqual(result, [])

    def test_multivariate_polynomial(self):
        """Test multivariate polynomial."""
        # Create symbols
        x, y = symbols("x y")

        # Create a multivariate polynomial: x^2 + 2*x*y + y^2
        poly = Poly(x**2 + 2 * x * y + y**2, x, y)

        # Test the function
        # The representation for multivariate polynomials is more complex
        # and depends on the internal implementation of SymPy
        result = custom_generatePolyList(poly)

        # We'll just check that we get a list
        self.assertIsInstance(result, list)
        # For multivariate polynomials, the elements might not be integers
        # but rather lists or other structures depending on SymPy's implementation

    def test_non_polynomial_input(self):
        """Test handling of inputs that are not Poly instances."""
        # Create a symbol
        x = symbols("x")

        # Test with expressions instead of polynomials
        with self.assertRaises(AttributeError):
            custom_generatePolyList(x + 1)

        with self.assertRaises(AttributeError):
            custom_generatePolyList(x**2)

        # Test with other types
        with self.assertRaises(AttributeError):
            custom_generatePolyList("not a polynomial")

        with self.assertRaises(AttributeError):
            custom_generatePolyList(42)


if __name__ == "__main__":
    unittest.main()
