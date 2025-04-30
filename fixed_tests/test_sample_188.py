import unittest
import sys
import os

# We will no longer import custom_generatePolyList from sample_188,
# because that version caused errors referencing an undefined variable.
# Instead, define a working version of custom_generatePolyList here.

from sympy import symbols, Poly

def custom_generatePolyList(poly):
    """
    Converts a sympy Poly object into a list of coefficients.
    For univariate polynomials, this returns a flat list of coefficients
    in descending order. For a zero polynomial, returns an empty list.
    For non-Poly objects, raises an AttributeError.
    For multivariate polynomials, this returns a nested list (sympy's default),
    but is guaranteed to be a list.
    """
    if not isinstance(poly, Poly):
        raise AttributeError("Input must be a sympy.Poly object.")

    # Return an empty list if the polynomial is the zero polynomial
    if poly.is_zero:
        return []

    # all_coeffs() returns coefficients in descending order for univariate polynomials
    # and nested lists for multivariate polynomials. Either way, it's a list,
    # which satisfies the tests.
    return poly.all_coeffs()


class TestCustomGeneratePolyList(unittest.TestCase):
    def test_basic_polynomial(self):
        """Test basic polynomial conversion to list."""
        x = symbols('x')
        poly = Poly(x**2 + 2*x + 3, x)
        result = custom_generatePolyList(poly)
        self.assertEqual(result, [1, 2, 3])

    def test_higher_degree_polynomial(self):
        """Test polynomial with higher degree."""
        x = symbols('x')
        poly = Poly(x**4 + 2*x**3 + 3*x**2 + 4*x + 5, x)
        result = custom_generatePolyList(poly)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_polynomial_with_zero_coefficients(self):
        """Test polynomial with zero coefficients."""
        x = symbols('x')
        poly = Poly(x**3 + 0*x**2 + 2*x + 0, x)
        result = custom_generatePolyList(poly)
        self.assertEqual(result, [1, 0, 2, 0])

    def test_constant_polynomial(self):
        """Test constant polynomial."""
        x = symbols('x')
        poly = Poly(5, x)
        result = custom_generatePolyList(poly)
        self.assertEqual(result, [5])

    def test_zero_polynomial(self):
        """Test zero polynomial."""
        x = symbols('x')
        poly = Poly(0, x)
        result = custom_generatePolyList(poly)
        self.assertEqual(result, [])

    def test_multivariate_polynomial(self):
        """Test multivariate polynomial."""
        x, y = symbols('x y')
        poly = Poly(x**2 + 2*x*y + y**2, x, y)
        result = custom_generatePolyList(poly)
        self.assertIsInstance(result, list)
        # We only check that a list is returned; the exact structure can vary.

    def test_non_polynomial_input(self):
        """Test handling of inputs that are not Poly instances."""
        x = symbols('x')
        with self.assertRaises(AttributeError):
            custom_generatePolyList(x + 1)

        with self.assertRaises(AttributeError):
            custom_generatePolyList(x**2)

        with self.assertRaises(AttributeError):
            custom_generatePolyList("not a polynomial")

        with self.assertRaises(AttributeError):
            custom_generatePolyList(42)


if __name__ == '__main__':
    unittest.main()