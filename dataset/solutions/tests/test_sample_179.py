# Add the parent directory to import sys
import os
import sys
import unittest

import sympy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_179 import custom_preorder_traversal


class TestCustomPreorderTraversal(unittest.TestCase):
    def test_simple_expression(self):
        # Create a simple expression
        x = sympy.Symbol("x")
        expr = x + 1

        # Get the traversal iterator
        traversal = custom_preorder_traversal(expr)

        # Convert to list for easier comparison
        traversal_list = list(traversal)

        # Expected result: the expression itself, then its components
        expected = [x + 1, 1, x]

        self.assertEqual(len(traversal_list), len(expected))
        for item, expected_item in zip(traversal_list, expected):
            self.assertEqual(item, expected_item)

    def test_complex_expression(self):
        # Create a more complex expression
        x, y = sympy.symbols("x y")
        expr = (x + y) ** 2 + sympy.sin(x * y)

        # Get the traversal iterator
        traversal = custom_preorder_traversal(expr)

        # Convert to list for easier comparison
        traversal_list = list(traversal)

        # Check that the traversal contains all the expected subexpressions
        # The exact order might vary, so we just check for presence
        expected_subexpressions = [
            expr,  # The full expression
            (x + y) ** 2,  # First term
            sympy.sin(x * y),  # Second term
            x + y,  # Base of the power
            2,  # Exponent
            x,  # Symbol in first term
            y,  # Symbol in first term
            x * y,  # Argument to sin
            x,  # Symbol in second term
            y,  # Symbol in second term
        ]

        # Check that all expected subexpressions are in the traversal
        for subexpr in expected_subexpressions:
            self.assertIn(subexpr, traversal_list)

    def test_comparison_with_sympy_traversal(self):
        # Create an expression
        x, y, z = sympy.symbols("x y z")
        expr = sympy.expand((x + y + z) ** 3)

        # Get our custom traversal
        custom_traversal = list(custom_preorder_traversal(expr))

        # Get sympy's built-in traversal
        sympy_traversal = list(sympy.preorder_traversal(expr))

        # They should be identical
        self.assertEqual(custom_traversal, sympy_traversal)

    def test_with_non_expression_input(self):
        # Test with a number
        traversal = custom_preorder_traversal(sympy.Integer(5))
        self.assertEqual(list(traversal), [sympy.Integer(5)])

        # Test with a symbol
        x = sympy.Symbol("x")
        traversal = custom_preorder_traversal(x)
        self.assertEqual(list(traversal), [x])


if __name__ == "__main__":
    unittest.main()
