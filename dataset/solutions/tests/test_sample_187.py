import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_187 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_187 import custom_function
from sympy import symbols, Eq, sin, cos, exp, log, sqrt, pi, I, S


class TestCustomFunction(unittest.TestCase):
    def test_basic_equality_conversion(self):
        """Test basic conversion of equality to expression."""
        # Create symbols
        x = symbols("x")

        # Create a simple equality
        eq = Eq(x, 5)

        # Test the function
        result = custom_function(eq)

        # Check that the result is x - 5
        self.assertEqual(result, x - 5)

        # Test with reversed sides
        eq_reversed = Eq(5, x)
        result_reversed = custom_function(eq_reversed)

        # Check that the result is 5 - x
        self.assertEqual(result_reversed, 5 - x)

    def test_symbolic_equations(self):
        """Test with symbolic equations."""
        # Create symbols
        x, y, z = symbols("x y z")

        # Create symbolic equations
        eq1 = Eq(x, y)
        eq2 = Eq(x + y, z)
        eq3 = Eq(x * y, z**2)

        # Test the function
        result1 = custom_function(eq1)
        result2 = custom_function(eq2)
        result3 = custom_function(eq3)

        # Check the results
        self.assertEqual(result1, x - y)
        self.assertEqual(result2, x + y - z)
        self.assertEqual(result3, x * y - z**2)

    def test_numeric_equations(self):
        """Test with numeric equations."""
        # Create symbols
        x = symbols("x")

        # Create numeric equations
        eq1 = Eq(x, 10)
        eq2 = Eq(2 * x, 20)
        eq3 = Eq(x**2, 100)

        # Test the function
        result1 = custom_function(eq1)
        result2 = custom_function(eq2)
        result3 = custom_function(eq3)

        # Check the results
        self.assertEqual(result1, x - 10)
        self.assertEqual(result2, 2 * x - 20)
        self.assertEqual(result3, x**2 - 100)

    def test_complex_expressions(self):
        """Test with complex mathematical expressions."""
        # Create symbols
        x, y = symbols("x y")

        # Create equations with complex expressions
        eq1 = Eq(x**2 + 2 * x * y + y**2, (x + y) ** 2)
        eq2 = Eq(sin(x) ** 2 + cos(x) ** 2, 1)
        eq3 = Eq(exp(x + y), exp(x) * exp(y))

        # Test the function
        result1 = custom_function(eq1)
        result2 = custom_function(eq2)
        result3 = custom_function(eq3)

        # Check the results
        self.assertEqual(result1, x**2 + 2 * x * y + y**2 - (x + y) ** 2)
        self.assertEqual(result2, sin(x) ** 2 + cos(x) ** 2 - 1)
        self.assertEqual(result3, exp(x + y) - exp(x) * exp(y))

        # These should all simplify to 0
        self.assertEqual(result1.simplify(), 0)
        self.assertEqual(result2.simplify(), 0)
        self.assertEqual(result3.simplify(), 0)

    def test_equations_with_functions(self):
        """Test with equations containing various functions."""
        # Create symbols
        x = symbols("x")

        # Create equations with functions
        eq1 = Eq(log(exp(x)), x)
        eq2 = Eq(sin(x) ** 2 + cos(x) ** 2, 1)
        eq3 = Eq(sqrt(x**2), abs(x))

        # Test the function
        result1 = custom_function(eq1)
        result2 = custom_function(eq2)
        result3 = custom_function(eq3)

        # Check the results
        self.assertEqual(result1, log(exp(x)) - x)
        self.assertEqual(result2, sin(x) ** 2 + cos(x) ** 2 - 1)
        self.assertEqual(result3, sqrt(x**2) - abs(x))

        # This should simplify to 0
        self.assertEqual(result2.simplify(), 0)

        # Note: log(exp(x)) doesn't always simplify to x in SymPy
        # depending on the domain, so we won't test that

    def test_non_equality_inputs(self):
        """Test handling of inputs that are not Equality instances."""
        # Create symbols
        x, y = symbols("x y")

        # Test with expressions instead of equalities
        with self.assertRaises(AttributeError):
            custom_function(x + y)

        with self.assertRaises(AttributeError):
            custom_function(x**2)

        with self.assertRaises(AttributeError):
            custom_function(sin(x))

        # Test with other types
        with self.assertRaises(AttributeError):
            custom_function("not an equality")

        with self.assertRaises(AttributeError):
            custom_function(42)

    def test_identical_sides_equation(self):
        """Test with equations where both sides are identical."""
        # Create symbols
        x, y = symbols("x y")

        # When both sides of an equation are identical, SymPy returns a BooleanTrue object
        # instead of an Equality object, which doesn't have lhs and rhs attributes
        # We'll test this behavior

        # Create equations with identical sides
        eq1 = Eq(x, x)
        eq2 = Eq(x + y, x + y)
        eq3 = Eq(sin(x), sin(x))

        # Check that they are all BooleanTrue
        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)

        # Since these are not Equality objects, they will raise AttributeError
        # when passed to custom_function
        with self.assertRaises(AttributeError):
            custom_function(eq1)

        with self.assertRaises(AttributeError):
            custom_function(eq2)

        with self.assertRaises(AttributeError):
            custom_function(eq3)

    def test_equations_with_multiple_variables(self):
        """Test with equations containing multiple variables."""
        # Create symbols
        x, y, z, t = symbols("x y z t")

        # Create equations with multiple variables
        eq1 = Eq(x + y, z + t)
        eq2 = Eq(x * y * z, t**3)
        eq3 = Eq(x**2 + y**2 + z**2, t**2)

        # Test the function
        result1 = custom_function(eq1)
        result2 = custom_function(eq2)
        result3 = custom_function(eq3)

        # Check the results
        self.assertEqual(result1, x + y - z - t)
        self.assertEqual(result2, x * y * z - t**3)
        self.assertEqual(result3, x**2 + y**2 + z**2 - t**2)

        # Test with substitution
        self.assertEqual(result1.subs({x: 1, y: 2, z: 3, t: 0}), 1 + 2 - 3 - 0)
        self.assertEqual(result2.subs({x: 2, y: 3, z: 4, t: 6}), 2 * 3 * 4 - 6**3)
        self.assertEqual(
            result3.subs({x: 1, y: 2, z: 2, t: 3}), 1**2 + 2**2 + 2**2 - 3**2
        )


if __name__ == "__main__":
    unittest.main()
