import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_199 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_199 import custom_divides


class TestCustomDivides(unittest.TestCase):
    def test_with_divisible_numbers(self):
        """Test custom_divides with divisible numbers."""
        # Test a range of divisible number pairs
        divisible_pairs = [
            (10, 2),  # 10 is divisible by 2
            (15, 3),  # 15 is divisible by 3
            (20, 4),  # 20 is divisible by 4
            (25, 5),  # 25 is divisible by 5
            (100, 10),  # 100 is divisible by 10
            (144, 12),  # 144 is divisible by 12
        ]
        for n, p in divisible_pairs:
            with self.subTest(n=n, p=p):
                self.assertTrue(custom_divides(n, p))

    def test_with_non_divisible_numbers(self):
        """Test custom_divides with non-divisible numbers."""
        # Test a range of non-divisible number pairs
        non_divisible_pairs = [
            (10, 3),  # 10 is not divisible by 3
            (15, 4),  # 15 is not divisible by 4
            (20, 7),  # 20 is not divisible by 7
            (25, 6),  # 25 is not divisible by 6
            (100, 11),  # 100 is not divisible by 11
            (144, 13),  # 144 is not divisible by 13
        ]
        for n, p in non_divisible_pairs:
            with self.subTest(n=n, p=p):
                self.assertFalse(custom_divides(n, p))

    def test_with_zero_as_dividend(self):
        """Test custom_divides with zero as the dividend."""
        # Zero is divisible by any non-zero number
        divisors = [1, 2, 3, 4, 5, 10, 100]
        for p in divisors:
            with self.subTest(p=p):
                self.assertTrue(custom_divides(0, p))

    def test_with_zero_as_divisor(self):
        """Test custom_divides with zero as the divisor."""
        # Division by zero raises a ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            custom_divides(10, 0)

        with self.assertRaises(ZeroDivisionError):
            custom_divides(0, 0)

    def test_with_negative_numbers(self):
        """Test custom_divides with negative numbers."""
        # Test with negative dividend
        self.assertTrue(custom_divides(-10, 2))  # -10 is divisible by 2
        self.assertFalse(custom_divides(-10, 3))  # -10 is not divisible by 3

        # Test with negative divisor
        self.assertTrue(custom_divides(10, -2))  # 10 is divisible by -2
        self.assertFalse(custom_divides(10, -3))  # 10 is not divisible by -3

        # Test with both negative
        self.assertTrue(custom_divides(-10, -2))  # -10 is divisible by -2
        self.assertFalse(custom_divides(-10, -3))  # -10 is not divisible by -3

    def test_with_large_numbers(self):
        """Test custom_divides with large numbers."""
        # Test with large divisible numbers
        self.assertTrue(custom_divides(1000000, 1000))  # 1000000 is divisible by 1000
        self.assertTrue(custom_divides(2**20, 2**10))  # 2^20 is divisible by 2^10

        # Test with large non-divisible numbers
        self.assertFalse(
            custom_divides(1000001, 1000)
        )  # 1000001 is not divisible by 1000
        self.assertFalse(
            custom_divides(2**20 + 1, 2**10)
        )  # 2^20 + 1 is not divisible by 2^10

    def test_with_non_integer_input(self):
        """Test custom_divides with non-integer input."""
        # The function expects integer inputs
        # Let's check how it handles non-integer inputs
        try:
            result = custom_divides(10.0, 2.0)
            # If we get here, the function accepted the input
            # Let's check what it returned for floats that are divisible
            self.assertTrue(result)
        except Exception as e:
            # If an exception is raised, it could be a TypeError
            self.assertIsInstance(e, TypeError)

        try:
            result = custom_divides(10.5, 2)
            # If we get here, the function accepted the input
            # Let's check what it returned for a float that is not divisible
            self.assertFalse(result)
        except Exception as e:
            # If an exception is raised, it could be a TypeError
            self.assertIsInstance(e, TypeError)

    def test_return_type(self):
        """Test that the return type is a boolean."""
        result = custom_divides(10, 2)
        self.assertIsInstance(result, bool)

        result = custom_divides(10, 3)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
