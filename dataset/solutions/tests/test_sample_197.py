import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_197 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_197 import custom_is_perfect_square


class TestCustomIsPerfectSquare(unittest.TestCase):
    def test_with_perfect_squares(self):
        """Test custom_is_perfect_square with perfect squares."""
        # Test a range of perfect squares
        perfect_squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
        for num in perfect_squares:
            with self.subTest(num=num):
                self.assertTrue(custom_is_perfect_square(num))

    def test_with_non_perfect_squares(self):
        """Test custom_is_perfect_square with non-perfect squares."""
        # Test a range of non-perfect squares
        non_perfect_squares = [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        for num in non_perfect_squares:
            with self.subTest(num=num):
                self.assertFalse(custom_is_perfect_square(num))

    def test_with_zero(self):
        """Test custom_is_perfect_square with zero."""
        # Zero is considered a perfect square (0^2 = 0)
        self.assertTrue(custom_is_perfect_square(0))

    def test_with_negative_numbers(self):
        """Test custom_is_perfect_square with negative numbers."""
        # Negative numbers are not perfect squares in the real number system
        negative_numbers = [-1, -4, -9, -16, -25]
        for num in negative_numbers:
            with self.subTest(num=num):
                self.assertFalse(custom_is_perfect_square(num))

    def test_with_large_numbers(self):
        """Test custom_is_perfect_square with large numbers."""
        # Test with large perfect squares
        large_perfect_squares = [10000, 1000000, 10000000000]  # 100^2, 1000^2, 100000^2
        for num in large_perfect_squares:
            with self.subTest(num=num):
                self.assertTrue(custom_is_perfect_square(num))

        # Test with large non-perfect squares
        large_non_perfect_squares = [10001, 1000001, 10000000001]
        for num in large_non_perfect_squares:
            with self.subTest(num=num):
                self.assertFalse(custom_is_perfect_square(num))

    def test_with_non_integer_input(self):
        """Test custom_is_perfect_square with non-integer input."""
        # The function expects an integer input
        # Let's check how it handles non-integer inputs
        try:
            result = custom_is_perfect_square(4.0)
            # If we get here, the function accepted the input
            # Let's check what it returned for a float that is a perfect square
            self.assertTrue(result)
        except Exception as e:
            # If an exception is raised, it could be a ValueError or TypeError
            self.assertTrue(isinstance(e, (ValueError, TypeError)))

        try:
            result = custom_is_perfect_square(4.5)
            # If we get here, the function accepted the input
            # Let's check what it returned for a float that is not a perfect square
            self.assertFalse(result)
        except Exception as e:
            # If an exception is raised, it could be a ValueError or TypeError
            self.assertTrue(isinstance(e, (ValueError, TypeError)))

    def test_with_edge_cases(self):
        """Test custom_is_perfect_square with edge cases."""
        # Test with 1, which is a perfect square (1^2 = 1)
        self.assertTrue(custom_is_perfect_square(1))

        # Test with very large perfect square
        # 2^30 = 1073741824, and (2^15)^2 = 1073741824
        self.assertTrue(custom_is_perfect_square(1073741824))

    def test_return_type(self):
        """Test that the return type is a boolean."""
        result = custom_is_perfect_square(4)
        self.assertIsInstance(result, bool)

        result = custom_is_perfect_square(5)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
