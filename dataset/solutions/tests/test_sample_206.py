import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_206 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_206 import custom_mobius
import sympy


class TestCustomMobius(unittest.TestCase):
    def test_basic_mobius_values(self):
        """Test custom_mobius with basic known values."""
        # Test the first 20 Möbius function values
        expected_values = [
            None,  # μ(0) is not defined (index 0 is not used)
            1,  # μ(1) = 1
            -1,  # μ(2) = -1
            -1,  # μ(3) = -1
            0,  # μ(4) = 0
            -1,  # μ(5) = -1
            1,  # μ(6) = 1
            -1,  # μ(7) = -1
            0,  # μ(8) = 0
            0,  # μ(9) = 0
            1,  # μ(10) = 1
            -1,  # μ(11) = -1
            0,  # μ(12) = 0
            -1,  # μ(13) = -1
            1,  # μ(14) = 1
            1,  # μ(15) = 1
            0,  # μ(16) = 0
            -1,  # μ(17) = -1
            0,  # μ(18) = 0
            -1,  # μ(19) = -1
            0,  # μ(20) = 0
        ]

        for n in range(1, 21):
            self.assertEqual(custom_mobius(n), expected_values[n], f"Failed for n={n}")

    def test_mobius_of_prime_numbers(self):
        """Test custom_mobius with prime numbers."""
        # For a prime number p, μ(p) = -1
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for prime in primes:
            self.assertEqual(custom_mobius(prime), -1, f"Failed for prime {prime}")

    def test_mobius_of_products_of_distinct_primes(self):
        """Test custom_mobius with products of distinct primes."""
        # For a product of k distinct primes, μ(n) = (-1)^k

        # Products of 2 distinct primes (k=2, so μ(n) = 1)
        products_2_primes = [
            6,  # 2 * 3
            10,  # 2 * 5
            14,  # 2 * 7
            15,  # 3 * 5
            21,  # 3 * 7
            22,  # 2 * 11
            26,  # 2 * 13
            33,  # 3 * 11
            35,  # 5 * 7
            38,  # 2 * 19
        ]
        for n in products_2_primes:
            self.assertEqual(custom_mobius(n), 1, f"Failed for n={n}")

        # Products of 3 distinct primes (k=3, so μ(n) = -1)
        products_3_primes = [
            30,  # 2 * 3 * 5
            42,  # 2 * 3 * 7
            66,  # 2 * 3 * 11
            70,  # 2 * 5 * 7
            105,  # 3 * 5 * 7
            110,  # 2 * 5 * 11
            154,  # 2 * 7 * 11
            165,  # 3 * 5 * 11
        ]
        for n in products_3_primes:
            self.assertEqual(custom_mobius(n), -1, f"Failed for n={n}")

        # Products of 4 distinct primes (k=4, so μ(n) = 1)
        products_4_primes = [
            210,  # 2 * 3 * 5 * 7
            330,  # 2 * 3 * 5 * 11
            462,  # 2 * 3 * 7 * 11
            546,  # 2 * 3 * 7 * 13
            770,  # 2 * 5 * 7 * 11
            858,  # 2 * 3 * 11 * 13
            1155,  # 3 * 5 * 7 * 11
        ]
        for n in products_4_primes:
            self.assertEqual(custom_mobius(n), 1, f"Failed for n={n}")

    def test_mobius_of_numbers_with_squared_factors(self):
        """Test custom_mobius with numbers that have squared prime factors."""
        # If n has a squared prime factor, μ(n) = 0
        squared_factors = [
            4,  # 2^2
            8,  # 2^3
            9,  # 3^2
            12,  # 2^2 * 3
            16,  # 2^4
            18,  # 2 * 3^2
            20,  # 2^2 * 5
            24,  # 2^3 * 3
            25,  # 5^2
            27,  # 3^3
            28,  # 2^2 * 7
            32,  # 2^5
            36,  # 2^2 * 3^2
            40,  # 2^3 * 5
            44,  # 2^2 * 11
            45,  # 3^2 * 5
            48,  # 2^4 * 3
            49,  # 7^2
            50,  # 2 * 5^2
            52,  # 2^2 * 13
            54,  # 2 * 3^3
            56,  # 2^3 * 7
            60,  # 2^2 * 3 * 5
            63,  # 3^2 * 7
            64,  # 2^6
            72,  # 2^3 * 3^2
            75,  # 3 * 5^2
            80,  # 2^4 * 5
            81,  # 3^4
            84,  # 2^2 * 3 * 7
            88,  # 2^3 * 11
            90,  # 2 * 3^2 * 5
            96,  # 2^5 * 3
            98,  # 2 * 7^2
            99,  # 3^2 * 11
            100,  # 2^2 * 5^2
        ]
        for n in squared_factors:
            self.assertEqual(custom_mobius(n), 0, f"Failed for n={n}")

    def test_mobius_multiplicative_property(self):
        """Test custom_mobius for the multiplicative property."""
        # If gcd(m, n) = 1, then μ(m*n) = μ(m) * μ(n)

        # Test with some relatively prime pairs
        # gcd(2, 3) = 1
        self.assertEqual(custom_mobius(2 * 3), custom_mobius(2) * custom_mobius(3))

        # gcd(2, 5) = 1
        self.assertEqual(custom_mobius(2 * 5), custom_mobius(2) * custom_mobius(5))

        # gcd(3, 5) = 1
        self.assertEqual(custom_mobius(3 * 5), custom_mobius(3) * custom_mobius(5))

        # gcd(2, 7) = 1
        self.assertEqual(custom_mobius(2 * 7), custom_mobius(2) * custom_mobius(7))

        # gcd(3, 7) = 1
        self.assertEqual(custom_mobius(3 * 7), custom_mobius(3) * custom_mobius(7))

        # gcd(5, 7) = 1
        self.assertEqual(custom_mobius(5 * 7), custom_mobius(5) * custom_mobius(7))

        # gcd(2, 11) = 1
        self.assertEqual(custom_mobius(2 * 11), custom_mobius(2) * custom_mobius(11))

        # gcd(3, 11) = 1
        self.assertEqual(custom_mobius(3 * 11), custom_mobius(3) * custom_mobius(11))

    def test_edge_cases(self):
        """Test custom_mobius with edge cases (0, 1)."""
        # μ(1) = 1 (by definition)
        self.assertEqual(custom_mobius(1), 1)

        # μ(0) is not mathematically defined
        # SymPy might raise an exception or return a specific value
        try:
            result = custom_mobius(0)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since μ(0) is not defined
            pass

    def test_negative_numbers(self):
        """Test custom_mobius with negative numbers."""
        # The Möbius function is not defined for negative numbers
        # SymPy might raise an exception or handle this appropriately

        # Test with some negative numbers
        try:
            result = custom_mobius(-1)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        try:
            result = custom_mobius(-5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

    def test_non_integer_inputs(self):
        """Test custom_mobius with non-integer inputs."""
        # The Möbius function is defined only for integers
        # SymPy should raise an exception or handle this appropriately

        # Test with float input
        try:
            result = custom_mobius(3.5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

        # Test with string input
        try:
            result = custom_mobius("3")
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass


if __name__ == "__main__":
    unittest.main()
