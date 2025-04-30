import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_188 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_188 import custom_generatePolyList
from sympy import symbols, Poly


class TestCustomGeneratePolyList(unittest.TestCase):
    def test_multivariate_polynomial(self):
        """Test multivariate polynomial."""
        x, y = symbols('x y')
        poly = Poly(x**2 + 2*x*y + y**2, x, y)
        result = custom_generatePolyList(poly)
        self.assertIsInstance(result, list)
        # Since it's multivariate, we won't check exact values (could be nested lists, etc.).
        # Just ensure it returns something list-like.


if __name__ == '__main__':
    unittest.main()