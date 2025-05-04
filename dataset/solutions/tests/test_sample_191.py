import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_191 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_191 import custom_symbol
from sympy import Indexed, Symbol, IndexedBase


class TestCustomSymbol(unittest.TestCase):
    def test_basic_indexed_symbol_returns_correct_set(self):
        """Test that a basic Indexed symbol returns the correct set of free symbols."""
        # Create a basic Indexed symbol
        i = Symbol("i")
        A = IndexedBase("A")
        indexed_expr = A[i]

        # Get the free symbols
        result = custom_symbol(indexed_expr)

        # Check that the result contains the expected symbols
        self.assertIn(i, result)
        # The result should also contain the base symbol A and the indexed expression itself
        self.assertTrue(any(s.name == "A" for s in result))
        # The set should have at least 3 elements (i, A, and A[i])
        self.assertGreaterEqual(len(result), 3)

    def test_multiple_free_symbols_in_indexed(self):
        """Test that an Indexed with multiple free symbols returns all of them."""
        # Create an Indexed with multiple free symbols
        i = Symbol("i")
        j = Symbol("j")
        A = IndexedBase("A")
        indexed_expr = A[i, j]

        # Get the free symbols
        result = custom_symbol(indexed_expr)

        # Check that the result contains all expected symbols
        self.assertIn(i, result)
        self.assertIn(j, result)
        # The result should also contain the base symbol A and the indexed expression itself
        self.assertTrue(any(s.name == "A" for s in result))
        # The set should have at least 4 elements (i, j, A, and A[i, j])
        self.assertGreaterEqual(len(result), 4)

    def test_indexed_with_no_free_symbols(self):
        """Test that an Indexed with numeric indices returns the base symbol."""
        # Create an Indexed with numeric indices
        A = IndexedBase("A")
        indexed_expr = A[1, 2]  # Using integers instead of symbols

        # Get the free symbols
        result = custom_symbol(indexed_expr)

        # Check that the result contains the base symbol
        self.assertTrue(any(s.name == "A" for s in result))
        # The set should have at least 2 elements (A and A[1, 2])
        self.assertGreaterEqual(len(result), 2)

    def test_return_type_is_a_set(self):
        """Test that the return type is a set."""
        # Create a basic Indexed symbol
        i = Symbol("i")
        A = IndexedBase("A")
        indexed_expr = A[i]

        # Get the free symbols
        result = custom_symbol(indexed_expr)

        # Check that the result is a set
        self.assertIsInstance(result, set)

    def test_return_set_contains_symbol_objects(self):
        """Test that the returned set contains Symbol objects."""
        # Create a basic Indexed symbol
        i = Symbol("i")
        A = IndexedBase("A")
        indexed_expr = A[i]

        # Get the free symbols
        result = custom_symbol(indexed_expr)

        # Check that the elements in the result are valid SymPy objects
        for symbol in result:
            # The set may contain Symbol, IndexedBase, or Indexed objects
            self.assertTrue(isinstance(symbol, (Symbol, IndexedBase, Indexed)))

    def test_complex_indexed_expression(self):
        """Test that a complex Indexed expression returns all free symbols."""
        # Create a complex Indexed expression
        i = Symbol("i")
        j = Symbol("j")
        k = Symbol("k")
        A = IndexedBase("A")
        B = IndexedBase("B")

        # A nested indexed expression: A[i, B[j, k]]
        inner_indexed = B[j, k]
        outer_indexed = A[i, inner_indexed]

        # Get the free symbols
        result = custom_symbol(outer_indexed)

        # Check that the result contains all expected symbols
        self.assertIn(i, result)
        self.assertIn(j, result)
        self.assertIn(k, result)
        # The result should also contain the base symbols A and B
        self.assertTrue(any(s.name == "A" for s in result))
        self.assertTrue(any(s.name == "B" for s in result))
        # The set should have at least 7 elements (i, j, k, A, B, B[j, k], A[i, B[j, k]])
        self.assertGreaterEqual(len(result), 7)
        # The exact number might depend on how sympy handles nested indexed expressions


if __name__ == "__main__":
    unittest.main()
