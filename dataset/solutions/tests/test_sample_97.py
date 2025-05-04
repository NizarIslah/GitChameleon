import unittest
import operator
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_97 import accumulate_functional


class TestAccumulateFunctional(unittest.TestCase):
    def test_accumulate_with_addition(self):
        """Test accumulate_functional with addition operator."""
        result = accumulate_functional([1, 2, 3, 4], operator.add)
        self.assertEqual(result, [1, 3, 6, 10])

    def test_accumulate_with_multiplication(self):
        """Test accumulate_functional with multiplication operator."""
        result = accumulate_functional([1, 2, 3, 4], operator.mul)
        self.assertEqual(result, [1, 2, 6, 24])

    def test_accumulate_with_custom_function(self):
        """Test accumulate_functional with a custom function."""

        def max_func(a, b):
            return max(a, b)

        result = accumulate_functional([1, 5, 3, 8, 2], max_func)
        self.assertEqual(result, [1, 5, 5, 8, 8])

    def test_accumulate_with_empty_list(self):
        """Test accumulate_functional with an empty list."""
        result = accumulate_functional([], operator.add)
        self.assertEqual(result, [])

    def test_accumulate_with_single_element(self):
        """Test accumulate_functional with a single element list."""
        result = accumulate_functional([42], operator.add)
        self.assertEqual(result, [42])

    def test_accumulate_with_strings(self):
        """Test accumulate_functional with string concatenation."""
        result = accumulate_functional(["a", "b", "c"], operator.add)
        self.assertEqual(result, ["a", "ab", "abc"])


if __name__ == "__main__":
    unittest.main()
