import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_205 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_205 import custom_totient
import sympy


class TestCustomTotient(unittest.TestCase):
    def test_basic_totient_values(self):
        """Test custom_totient with basic known values."""
        # Test some known totient values
        self.assertEqual(custom_totient(1), 1)  # φ(1) = 1
        self.assertEqual(custom_totient(2), 1)  # φ(2) = 1
        self.assertEqual(custom_totient(3), 2)  # φ(3) = 2
        self.assertEqual(custom_totient(4), 2)  # φ(4) = 2
        self.assertEqual(custom_totient(5), 4)  # φ(5) = 4
        self.assertEqual(custom_totient(6), 2)  # φ(6) = 2
        self.assertEqual(custom_totient(7), 6)  # φ(7) = 6
        self.assertEqual(custom_totient(8), 4)  # φ(8) = 4
        self.assertEqual(custom_totient(9), 6)  # φ(9) = 6
        self.assertEqual(custom_totient(10), 4)  # φ(10) = 4
        self.assertEqual(custom_totient(12), 4)  # φ(12) = 4
        self.assertEqual(custom_totient(24), 8)  # φ(24) = 8
        self.assertEqual(custom_totient(36), 12)  # φ(36) = 12

    def test_totient_of_prime_numbers(self):
        """Test custom_totient with prime numbers."""
        # For a prime number p, φ(p) = p - 1
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for prime in primes:
            self.assertEqual(
                custom_totient(prime), prime - 1, f"Failed for prime {prime}"
            )

    def test_totient_of_prime_powers(self):
        """Test custom_totient with powers of primes."""
        # For a prime power p^k, φ(p^k) = p^k - p^(k-1) = p^k * (1 - 1/p)

        # Test powers of 2
        self.assertEqual(
            custom_totient(2**1), 2**1 - 2 ** (1 - 1)
        )  # φ(2) = 2 - 1 = 1
        self.assertEqual(
            custom_totient(2**2), 2**2 - 2 ** (2 - 1)
        )  # φ(4) = 4 - 2 = 2
        self.assertEqual(
            custom_totient(2**3), 2**3 - 2 ** (3 - 1)
        )  # φ(8) = 8 - 4 = 4
        self.assertEqual(
            custom_totient(2**4), 2**4 - 2 ** (4 - 1)
        )  # φ(16) = 16 - 8 = 8
        self.assertEqual(
            custom_totient(2**5), 2**5 - 2 ** (5 - 1)
        )  # φ(32) = 32 - 16 = 16

        # Test powers of 3
        self.assertEqual(
            custom_totient(3**1), 3**1 - 3 ** (1 - 1)
        )  # φ(3) = 3 - 1 = 2
        self.assertEqual(
            custom_totient(3**2), 3**2 - 3 ** (2 - 1)
        )  # φ(9) = 9 - 3 = 6
        self.assertEqual(
            custom_totient(3**3), 3**3 - 3 ** (3 - 1)
        )  # φ(27) = 27 - 9 = 18
        self.assertEqual(
            custom_totient(3**4), 3**4 - 3 ** (4 - 1)
        )  # φ(81) = 81 - 27 = 54

        # Test powers of 5
        self.assertEqual(
            custom_totient(5**1), 5**1 - 5 ** (1 - 1)
        )  # φ(5) = 5 - 1 = 4
        self.assertEqual(
            custom_totient(5**2), 5**2 - 5 ** (2 - 1)
        )  # φ(25) = 25 - 5 = 20
        self.assertEqual(
            custom_totient(5**3), 5**3 - 5 ** (3 - 1)
        )  # φ(125) = 125 - 25 = 100

    def test_totient_of_composite_numbers(self):
        """Test custom_totient with composite numbers."""
        # Test some composite numbers with known totient values
        self.assertEqual(custom_totient(6), 2)  # φ(6) = 2
        self.assertEqual(custom_totient(10), 4)  # φ(10) = 4
        self.assertEqual(custom_totient(12), 4)  # φ(12) = 4
        self.assertEqual(custom_totient(15), 8)  # φ(15) = 8
        self.assertEqual(custom_totient(20), 8)  # φ(20) = 8
        self.assertEqual(custom_totient(30), 8)  # φ(30) = 8
        self.assertEqual(custom_totient(36), 12)  # φ(36) = 12
        self.assertEqual(custom_totient(60), 16)  # φ(60) = 16
        self.assertEqual(custom_totient(100), 40)  # φ(100) = 40
        self.assertEqual(custom_totient(144), 48)  # φ(144) = 48

    def test_totient_multiplicative_property(self):
        """Test custom_totient for the multiplicative property."""
        # If gcd(m, n) = 1, then φ(m*n) = φ(m) * φ(n)

        # Test with some relatively prime pairs
        # gcd(3, 4) = 1
        self.assertEqual(custom_totient(3 * 4), custom_totient(3) * custom_totient(4))

        # gcd(5, 8) = 1
        self.assertEqual(custom_totient(5 * 8), custom_totient(5) * custom_totient(8))

        # gcd(7, 10) = 1
        self.assertEqual(custom_totient(7 * 10), custom_totient(7) * custom_totient(10))

        # gcd(9, 16) = 1
        self.assertEqual(custom_totient(9 * 16), custom_totient(9) * custom_totient(16))

        # gcd(11, 20) = 1
        self.assertEqual(
            custom_totient(11 * 20), custom_totient(11) * custom_totient(20)
        )

    def test_edge_cases(self):
        """Test custom_totient with edge cases (0, 1)."""
        # φ(1) = 1 (by definition)
        self.assertEqual(custom_totient(1), 1)

        # φ(0) is not mathematically defined
        # SymPy might raise an exception or return a specific value
        try:
            result = custom_totient(0)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since φ(0) is not defined
            pass

    def test_negative_numbers(self):
        """Test custom_totient with negative numbers."""
        # The totient function is not defined for negative numbers
        # SymPy might raise an exception or handle this appropriately

        # Test with some negative numbers
        try:
            result = custom_totient(-1)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        try:
            result = custom_totient(-5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

    def test_non_integer_inputs(self):
        """Test custom_totient with non-integer inputs."""
        # The totient function is defined only for integers
        # SymPy should raise an exception or handle this appropriately

        # Test with float input
        try:
            result = custom_totient(3.5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        # Test with string input
        try:
            result = custom_totient("3")
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass


if __name__ == "__main__":
    unittest.main()
