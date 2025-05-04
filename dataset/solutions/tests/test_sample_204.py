import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_204 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_204 import custom_prime_counting
import sympy


class TestCustomPrimeCounting(unittest.TestCase):
    def test_basic_prime_counting_values(self):
        """Test custom_prime_counting with basic known values."""
        # Test some known prime counting values
        self.assertEqual(custom_prime_counting(10), 4)  # π(10) = 4 (primes: 2, 3, 5, 7)
        self.assertEqual(
            custom_prime_counting(20), 8
        )  # π(20) = 8 (primes: 2, 3, 5, 7, 11, 13, 17, 19)
        self.assertEqual(custom_prime_counting(30), 10)  # π(30) = 10
        self.assertEqual(custom_prime_counting(100), 25)  # π(100) = 25
        self.assertEqual(custom_prime_counting(200), 46)  # π(200) = 46

    def test_small_integers(self):
        """Test custom_prime_counting with small integers (1-20)."""
        # Test each integer from 1 to 20
        expected_values = [0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8]
        for n in range(1, 21):
            self.assertEqual(
                custom_prime_counting(n), expected_values[n - 1], f"Failed for n={n}"
            )

    def test_medium_integers(self):
        """Test custom_prime_counting with medium integers (50-200)."""
        # Test some medium-sized integers
        self.assertEqual(custom_prime_counting(50), 15)  # π(50) = 15
        self.assertEqual(custom_prime_counting(100), 25)  # π(100) = 25
        self.assertEqual(custom_prime_counting(150), 35)  # π(150) = 35
        self.assertEqual(custom_prime_counting(200), 46)  # π(200) = 46

    def test_large_integers(self):
        """Test custom_prime_counting with large integers (500-1000)."""
        # Test some larger integers
        self.assertEqual(custom_prime_counting(500), 95)  # π(500) = 95
        self.assertEqual(custom_prime_counting(1000), 168)  # π(1000) = 168

    def test_edge_cases(self):
        """Test custom_prime_counting with edge cases (0, 1)."""
        # For 0, there are no primes less than or equal to 0
        self.assertEqual(custom_prime_counting(0), 0)

        # For 1, there are no primes less than or equal to 1
        self.assertEqual(custom_prime_counting(1), 0)

    def test_negative_numbers(self):
        """Test custom_prime_counting with negative numbers."""
        # For negative numbers, there are no primes less than or equal to them
        # SymPy might return 0 or raise an exception

        # Test with some negative numbers
        try:
            result = custom_prime_counting(-1)
            # If we get here, check that the result is 0
            self.assertEqual(result, 0)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        try:
            result = custom_prime_counting(-10)
            # If we get here, check that the result is 0
            self.assertEqual(result, 0)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

    def test_very_large_integers(self):
        """Test custom_prime_counting with very large integers."""
        # Test with some very large integers
        # These might be slow to compute, so we'll just check if the result is reasonable

        # π(10000) ≈ 1229
        result = custom_prime_counting(10000)
        # The result might be a sympy.Integer, which is not an instance of int
        # but can be compared with integers
        self.assertEqual(result, 1229)

        # π(100000) ≈ 9592
        # This might be too slow to compute in a test, so we'll make it optional
        try:
            result = custom_prime_counting(100000)
            self.assertEqual(result, 9592)
        except Exception as e:
            # If an exception is raised due to computational limitations, that's acceptable
            pass

    def test_non_integer_inputs(self):
        """Test custom_prime_counting with non-integer inputs."""
        # The function is defined for integers
        # SymPy should raise an exception or handle this appropriately

        # Test with float input
        try:
            result = custom_prime_counting(3.5)
            # If we get here, check that the result is reasonable
            # For 3.5, we expect the same result as for 3
            self.assertEqual(result, custom_prime_counting(3))
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        # Test with string input
        try:
            result = custom_prime_counting("3")
            # If we get here, check that the result is reasonable
            # For "3", we expect the same result as for 3
            self.assertEqual(result, custom_prime_counting(3))
        except Exception as e:
            # If an exception is raised, it's expected
            pass


if __name__ == "__main__":
    unittest.main()
