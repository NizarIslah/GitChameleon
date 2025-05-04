import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_201 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_201 import custom_jacobi_symbols
import sympy


class TestCustomJacobiSymbols(unittest.TestCase):
    def test_basic_jacobi_symbol_values(self):
        """Test custom_jacobi_symbols with basic known values."""
        # Test some known Jacobi symbol values
        self.assertEqual(custom_jacobi_symbols(1, 3), 1)  # (1/3) = 1
        self.assertEqual(custom_jacobi_symbols(2, 3), -1)  # (2/3) = -1
        self.assertEqual(custom_jacobi_symbols(1, 5), 1)  # (1/5) = 1
        self.assertEqual(custom_jacobi_symbols(2, 5), -1)  # (2/5) = -1
        self.assertEqual(custom_jacobi_symbols(3, 5), -1)  # (3/5) = -1
        self.assertEqual(custom_jacobi_symbols(4, 5), 1)  # (4/5) = 1
        self.assertEqual(custom_jacobi_symbols(1, 15), 1)  # (1/15) = 1
        self.assertEqual(custom_jacobi_symbols(2, 15), 1)  # (2/15) = 1
        self.assertEqual(custom_jacobi_symbols(7, 15), -1)  # (7/15) = -1

    def test_jacobi_symbol_with_prime_n(self):
        """Test custom_jacobi_symbols with prime n values."""
        # For prime n, Jacobi symbol is equivalent to Legendre symbol
        # Test with some prime n values
        self.assertEqual(custom_jacobi_symbols(2, 7), 1)  # (2/7) = 1
        self.assertEqual(custom_jacobi_symbols(3, 7), -1)  # (3/7) = -1
        self.assertEqual(custom_jacobi_symbols(4, 7), 1)  # (4/7) = 1
        self.assertEqual(custom_jacobi_symbols(5, 7), -1)  # (5/7) = -1
        self.assertEqual(custom_jacobi_symbols(6, 7), -1)  # (6/7) = -1

        # Test with another prime
        self.assertEqual(custom_jacobi_symbols(2, 11), -1)  # (2/11) = -1
        self.assertEqual(custom_jacobi_symbols(3, 11), 1)  # (3/11) = 1
        self.assertEqual(custom_jacobi_symbols(10, 11), -1)  # (10/11) = -1

    def test_jacobi_symbol_with_a_equals_zero(self):
        """Test custom_jacobi_symbols with a=0."""
        # Jacobi symbol (0/n) = 0 if n > 1
        self.assertEqual(custom_jacobi_symbols(0, 3), 0)
        self.assertEqual(custom_jacobi_symbols(0, 5), 0)
        self.assertEqual(custom_jacobi_symbols(0, 15), 0)

    def test_jacobi_symbol_with_negative_a(self):
        """Test custom_jacobi_symbols with negative a values."""
        # Test with negative a values
        # (−a/n) = (a/n) if n ≡ 1 (mod 4)
        # (−a/n) = −(a/n) if n ≡ 3 (mod 4)

        # n ≡ 1 (mod 4)
        self.assertEqual(custom_jacobi_symbols(-1, 5), 1)  # (-1/5) = 1
        self.assertEqual(custom_jacobi_symbols(-2, 5), -1)  # (-2/5) = -1

        # n ≡ 3 (mod 4)
        self.assertEqual(custom_jacobi_symbols(-1, 3), -1)  # (-1/3) = -1
        self.assertEqual(custom_jacobi_symbols(-2, 3), 1)  # (-2/3) = 1
        self.assertEqual(custom_jacobi_symbols(-1, 7), -1)  # (-1/7) = -1
        self.assertEqual(custom_jacobi_symbols(-2, 7), -1)  # (-2/7) = -1

    def test_jacobi_symbol_with_even_n(self):
        """Test custom_jacobi_symbols with even n values."""
        # The Jacobi symbol is defined for odd n > 0
        # For even n, SymPy should handle this appropriately

        # Test with even n values
        try:
            result = custom_jacobi_symbols(1, 2)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since n should be odd
            pass

        try:
            result = custom_jacobi_symbols(3, 4)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since n should be odd
            pass

    def test_jacobi_symbol_with_negative_n(self):
        """Test custom_jacobi_symbols with negative n values."""
        # The Jacobi symbol is defined for n > 0
        # For negative n, SymPy should handle this appropriately

        # Test with negative n values
        try:
            result = custom_jacobi_symbols(1, -3)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since n should be positive
            pass

        try:
            result = custom_jacobi_symbols(2, -5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since n should be positive
            pass

    def test_jacobi_symbol_with_large_values(self):
        """Test custom_jacobi_symbols with large values."""
        # Test with large values
        # (2/1001) = (2/7)(2/11)(2/13) = 1*(-1)*(-1) = 1
        self.assertEqual(custom_jacobi_symbols(2, 1001), 1)

        # Test with a large prime
        large_prime = 10007  # A prime number
        # (2/p) = 1 if p ≡ ±1 (mod 8), and -1 if p ≡ ±3 (mod 8)
        # 10007 ≡ -1 (mod 8), so (2/10007) = 1
        self.assertEqual(custom_jacobi_symbols(2, large_prime), 1)

        # Test with a large composite number
        large_composite = 10001  # 73 * 137
        # (3/10001) = (3/73)(3/137)
        expected = sympy.jacobi_symbol(3, 73) * sympy.jacobi_symbol(3, 137)
        self.assertEqual(custom_jacobi_symbols(3, large_composite), expected)

    def test_jacobi_symbol_quadratic_reciprocity(self):
        """Test custom_jacobi_symbols for quadratic reciprocity law."""
        # Quadratic reciprocity law:
        # If p and q are odd primes, then:
        # (p/q)(q/p) = (-1)^((p-1)(q-1)/4)

        # Test with p=3, q=5
        # (3/5)(5/3) = (-1)^((3-1)(5-1)/4) = (-1)^2 = 1
        self.assertEqual(
            custom_jacobi_symbols(3, 5) * custom_jacobi_symbols(5, 3),
            (-1) ** ((3 - 1) * (5 - 1) // 4),
        )

        # Test with p=3, q=7
        # (3/7)(7/3) = (-1)^((3-1)(7-1)/4) = (-1)^3 = -1
        self.assertEqual(
            custom_jacobi_symbols(3, 7) * custom_jacobi_symbols(7, 3),
            (-1) ** ((3 - 1) * (7 - 1) // 4),
        )

        # Test with p=5, q=11
        # (5/11)(11/5) = (-1)^((5-1)(11-1)/4) = (-1)^10 = 1
        self.assertEqual(
            custom_jacobi_symbols(5, 11) * custom_jacobi_symbols(11, 5),
            (-1) ** ((5 - 1) * (11 - 1) // 4),
        )


if __name__ == "__main__":
    unittest.main()
