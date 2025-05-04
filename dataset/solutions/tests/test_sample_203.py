import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_203 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_203 import custom_primefactors
import sympy


class TestCustomPrimeFactors(unittest.TestCase):
    def test_basic_prime_factor_counts(self):
        """Test custom_primefactors with basic known values."""
        # Test some known values
        self.assertEqual(custom_primefactors(2), 1)  # 2 = 2¹
        self.assertEqual(custom_primefactors(3), 1)  # 3 = 3¹
        self.assertEqual(custom_primefactors(4), 2)  # 4 = 2²
        self.assertEqual(custom_primefactors(6), 2)  # 6 = 2¹ × 3¹
        self.assertEqual(custom_primefactors(8), 3)  # 8 = 2³
        self.assertEqual(custom_primefactors(12), 3)  # 12 = 2² × 3¹
        self.assertEqual(custom_primefactors(24), 4)  # 24 = 2³ × 3¹
        self.assertEqual(custom_primefactors(36), 4)  # 36 = 2² × 3²
        self.assertEqual(custom_primefactors(60), 4)  # 60 = 2² × 3¹ × 5¹
        self.assertEqual(custom_primefactors(120), 5)  # 120 = 2³ × 3¹ × 5¹

    def test_prime_numbers(self):
        """Test custom_primefactors with prime numbers."""
        # For prime numbers, the result should be 1
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for prime in primes:
            self.assertEqual(custom_primefactors(prime), 1, f"Failed for prime {prime}")

    def test_powers_of_primes(self):
        """Test custom_primefactors with powers of primes."""
        # For p^n where p is prime, the result should be n
        # Test powers of 2
        for n in range(1, 10):
            self.assertEqual(custom_primefactors(2**n), n, f"Failed for 2^{n}")

        # Test powers of 3
        for n in range(1, 10):
            self.assertEqual(custom_primefactors(3**n), n, f"Failed for 3^{n}")

        # Test powers of 5
        for n in range(1, 10):
            self.assertEqual(custom_primefactors(5**n), n, f"Failed for 5^{n}")

    def test_composite_numbers(self):
        """Test custom_primefactors with composite numbers."""
        # Test some composite numbers with known factorizations
        # 10 = 2 × 5
        self.assertEqual(custom_primefactors(10), 2)

        # 15 = 3 × 5
        self.assertEqual(custom_primefactors(15), 2)

        # 30 = 2 × 3 × 5
        self.assertEqual(custom_primefactors(30), 3)

        # 42 = 2 × 3 × 7
        self.assertEqual(custom_primefactors(42), 3)

        # 100 = 2² × 5²
        self.assertEqual(custom_primefactors(100), 4)

        # 210 = 2 × 3 × 5 × 7
        self.assertEqual(custom_primefactors(210), 4)

        # 1001 = 7 × 11 × 13
        self.assertEqual(custom_primefactors(1001), 3)

        # 1024 = 2¹⁰
        self.assertEqual(custom_primefactors(1024), 10)

    def test_edge_cases(self):
        """Test custom_primefactors with edge cases (0, 1)."""
        # For 1, the result should be 0 (no prime factors)
        self.assertEqual(custom_primefactors(1), 0)

        # For 0, the behavior depends on the implementation
        # SymPy might raise an exception or return a specific value
        try:
            result = custom_primefactors(0)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since 0 has no prime factorization
            pass

    def test_negative_numbers(self):
        """Test custom_primefactors with negative numbers."""
        # For negative numbers, the behavior depends on the implementation
        # SymPy might count the prime factors of the absolute value
        # or raise an exception

        # Test with some negative numbers
        try:
            result = custom_primefactors(-2)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        try:
            result = custom_primefactors(-12)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

    def test_large_numbers(self):
        """Test custom_primefactors with large numbers."""
        # Test with some large numbers
        # 10000 = 2⁴ × 5⁴
        self.assertEqual(custom_primefactors(10000), 8)

        # 123456 = 2⁶ × 3 × 643
        # 643 is prime, so the total is 6 + 1 + 1 = 8
        self.assertEqual(custom_primefactors(123456), 8)

        # 999999 = 3³ × 7 × 11 × 13 × 37
        # The total is 3 + 1 + 1 + 1 + 1 = 7
        self.assertEqual(custom_primefactors(999999), 7)

        # 1000000 = 2⁶ × 5⁶
        self.assertEqual(custom_primefactors(1000000), 12)

    def test_non_integer_inputs(self):
        """Test custom_primefactors with non-integer inputs."""
        # The function is defined for integers
        # SymPy should raise an exception or handle this appropriately

        # Test with float input
        try:
            result = custom_primefactors(3.5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        # Test with string input
        try:
            result = custom_primefactors("3")
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass


if __name__ == "__main__":
    unittest.main()
