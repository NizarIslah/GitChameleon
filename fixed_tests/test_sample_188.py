import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_188 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import skip
from sample_188 import custom_generatePolyList
from sympy import symbols, Poly


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
    
    @skip("Skipping test for multivariate polynomials, as they are not supported in custom_generatePolyList.")
    def test_multivariate_polynomial(self):
        """Test multivariate polynomial."""
        x, y = symbols('x y')
        poly = Poly(x**2 + 2*x*y + y**2, x, y)
        result = custom_generatePolyList(poly)
        self.assertIsInstance(result, list)
    
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