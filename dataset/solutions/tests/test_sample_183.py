import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_183 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_183 import custom_check_carmichael
from sympy.ntheory.factor_ import is_carmichael


class TestCustomCheckCarmichael(unittest.TestCase):
    def test_carmichael_numbers(self):
        """Test with known Carmichael numbers."""
        # The first few Carmichael numbers are 561, 1105, 1729, 2465, 2821, 6601
        carmichael_numbers = [561, 1105, 1729, 2465, 2821, 6601]

        for num in carmichael_numbers:
            with self.subTest(num=num):
                self.assertTrue(custom_check_carmichael(num))

    def test_non_carmichael_numbers(self):
        """Test with numbers that are not Carmichael numbers."""
        # Some non-Carmichael numbers
        non_carmichael_numbers = [1, 2, 4, 10, 100, 1000, 562, 1106]

        for num in non_carmichael_numbers:
            with self.subTest(num=num):
                self.assertFalse(custom_check_carmichael(num))

    def test_negative_numbers(self):
        """Test with negative numbers, which should not be Carmichael numbers."""
        negative_numbers = [-1, -561, -1729]

        for num in negative_numbers:
            with self.subTest(num=num):
                self.assertFalse(custom_check_carmichael(num))

    def test_function_matches_sympy_implementation(self):
        """Test that our function matches the sympy implementation."""
        test_numbers = [1, 2, 561, 1105, 1729, 2465, 2821, 6601, 10, 100]

        for num in test_numbers:
            with self.subTest(num=num):
                self.assertEqual(custom_check_carmichael(num), is_carmichael(num))


if __name__ == "__main__":
    unittest.main()
