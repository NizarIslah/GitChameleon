import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_196 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_196 import custom_use
from sympy import symbols, sin, cos, exp, pi, sqrt, Function, Derivative, Integral


class TestCustomUse(unittest.TestCase):
    def test_with_simple_expression(self):
        """Test custom_use with a simple expression."""
        x = symbols("x")
        expr = x + 1
        result = custom_use(expr)
        # The simple expression is already evaluated, so it should remain the same
        self.assertEqual(result, expr)

    def test_with_complex_expression(self):
        """Test custom_use with a complex expression."""
        x, y = symbols("x y")
        expr = (x + y) ** 2 + x * y
        result = custom_use(expr)
        # The expression is already in simplified form
        self.assertEqual(result, expr)

    def test_with_symbolic_expression(self):
        """Test custom_use with a symbolic expression."""
        x, y, z = symbols("x y z")
        expr = x**2 + y**2 + z**2
        result = custom_use(expr)
        self.assertEqual(result, expr)

    def test_with_unevaluated_functions(self):
        """Test custom_use with unevaluated functions."""
        x = symbols("x")
        # Create an unevaluated function
        f = Function("f")
        expr = f(x) + f(x + 1)
        result = custom_use(expr)
        # Since f is just a symbol, doit() won't change it
        self.assertEqual(result, expr)

    def test_with_trigonometric_expressions(self):
        """Test custom_use with trigonometric expressions."""
        x = symbols("x")
        # sin(pi) should evaluate to 0 when doit() is called
        expr = sin(pi) + cos(0)
        result = custom_use(expr)
        # After evaluation: sin(pi) = 0, cos(0) = 1
        self.assertEqual(result, 1)

    def test_with_non_expression_input(self):
        """Test custom_use with non-expression input."""
        # Let's check if the function handles non-expression input
        # If it doesn't raise an error, we'll verify the behavior
        try:
            result = custom_use(5)
            # If we get here, the function accepted the input
            # Let's check what it returned
            self.assertIsNotNone(result)
        except Exception as e:
            # If an exception is raised, it should be a TypeError
            self.assertIsInstance(e, TypeError)

    def test_return_type(self):
        """Test that the return type is a sympy expression."""
        x = symbols("x")
        expr = x**2
        result = custom_use(expr)
        from sympy import Expr

        self.assertIsInstance(result, Expr)

    def test_with_nested_expressions(self):
        """Test custom_use with nested expressions that can be evaluated."""
        x = symbols("x")
        # sqrt(4) should evaluate to 2, exp(0) should evaluate to 1
        expr = x**2 + sqrt(4) + exp(0)
        result = custom_use(expr)
        # After evaluation: x**2 + 2 + 1 = x**2 + 3
        expected = x**2 + 3
        self.assertEqual(result, expected)

    def test_with_derivatives_and_integrals(self):
        """Test custom_use with derivatives and integrals."""
        x = symbols("x")
        # Create a derivative that can be evaluated
        expr = Derivative(x**2, x)
        result = custom_use(expr)
        # After evaluation: d/dx(x^2) = 2*x
        expected = 2 * x
        self.assertEqual(result, expected)

        # Create an integral that can be evaluated
        expr = Integral(x, (x, 0, 1))
        result = custom_use(expr)
        # After evaluation: ∫x dx from 0 to 1 = 1/2
        # Use sympy's S(1)/2 to ensure we get the exact symbolic fraction
        from sympy import S

        expected = S(1) / 2
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
