import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_207 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_207 import custom_legendre
import sympy


class TestCustomLegendre(unittest.TestCase):
    def test_basic_legendre_values(self):
        """Test custom_legendre with basic known values."""
        # Test some known Legendre symbol values
        test_cases = [
            # (a, n, expected_result)
            (1, 3, 1),  # (1|3) = 1 (1 is a quadratic residue modulo 3)
            (2, 3, -1),  # (2|3) = -1 (2 is not a quadratic residue modulo 3)
            (1, 5, 1),  # (1|5) = 1
            (2, 5, -1),  # (2|5) = -1
            (3, 5, -1),  # (3|5) = -1
            (4, 5, 1),  # (4|5) = 1
            (1, 7, 1),  # (1|7) = 1
            (2, 7, 1),  # (2|7) = 1
            (3, 7, -1),  # (3|7) = -1
            (4, 7, 1),  # (4|7) = 1
            (5, 7, -1),  # (5|7) = -1
            (6, 7, -1),  # (6|7) = -1
        ]

        for a, n, expected in test_cases:
            self.assertEqual(custom_legendre(a, n), expected, f"Failed for ({a}|{n})")

    def test_legendre_properties(self):
        """Test properties of the Legendre symbol."""
        # Property 1: (a|p) = (a mod p|p)
        primes = [3, 5, 7, 11, 13, 17, 19, 23]
        for p in primes:
            a = 25  # 25 mod 7 = 4
            self.assertEqual(
                custom_legendre(a, p),
                custom_legendre(a % p, p),
                f"Failed property (a|p) = (a mod p|p) for a={a}, p={p}",
            )

        # Property 2: (1|p) = 1 for any odd prime p
        for p in primes:
            self.assertEqual(
                custom_legendre(1, p), 1, f"Failed property (1|p) = 1 for p={p}"
            )

        # Property 3: (ab|p) = (a|p)(b|p) for any odd prime p
        for p in primes:
            a, b = 2, 3
            self.assertEqual(
                custom_legendre(a * b, p),
                custom_legendre(a, p) * custom_legendre(b, p),
                f"Failed property (ab|p) = (a|p)(b|p) for a={a}, b={b}, p={p}",
            )

    def test_quadratic_residues(self):
        """Test Legendre symbol for quadratic residues."""
        # For prime p, if a is a quadratic residue modulo p, then (a|p) = 1
        # A number a is a quadratic residue modulo p if there exists x such that x^2 ≡ a (mod p)

        # Quadratic residues modulo 7 are 1, 2, 4 (since 1^2 ≡ 1, 3^2 ≡ 2, 2^2 ≡ 4 (mod 7))
        self.assertEqual(custom_legendre(1, 7), 1)
        self.assertEqual(custom_legendre(2, 7), 1)
        self.assertEqual(custom_legendre(4, 7), 1)

        # Non-quadratic residues modulo 7 are 3, 5, 6
        self.assertEqual(custom_legendre(3, 7), -1)
        self.assertEqual(custom_legendre(5, 7), -1)
        self.assertEqual(custom_legendre(6, 7), -1)

        # Quadratic residues modulo 11 are 1, 3, 4, 5, 9 (since 1^2 ≡ 1, 5^2 ≡ 3, 2^2 ≡ 4, 4^2 ≡ 5, 3^2 ≡ 9 (mod 11))
        self.assertEqual(custom_legendre(1, 11), 1)
        self.assertEqual(custom_legendre(3, 11), 1)
        self.assertEqual(custom_legendre(4, 11), 1)
        self.assertEqual(custom_legendre(5, 11), 1)
        self.assertEqual(custom_legendre(9, 11), 1)

    def test_legendre_law_of_quadratic_reciprocity(self):
        """Test the law of quadratic reciprocity."""
        # Law of quadratic reciprocity: For distinct odd primes p and q,
        # (p|q)(q|p) = (-1)^((p-1)/2 * (q-1)/2)

        prime_pairs = [(3, 5), (3, 7), (5, 7), (5, 11), (7, 11), (7, 13), (11, 13)]

        for p, q in prime_pairs:
            expected = (-1) ** ((p - 1) // 2 * (q - 1) // 2)
            actual = custom_legendre(p, q) * custom_legendre(q, p)
            self.assertEqual(
                actual,
                expected,
                f"Failed law of quadratic reciprocity for p={p}, q={q}",
            )

    def test_edge_cases(self):
        """Test edge cases for the Legendre symbol."""
        # When a is divisible by p, (a|p) = 0
        primes = [3, 5, 7, 11, 13]
        for p in primes:
            self.assertEqual(custom_legendre(p, p), 0, f"Failed for ({p}|{p})")
            self.assertEqual(custom_legendre(2 * p, p), 0, f"Failed for ({2*p}|{p})")

        # When a is negative
        # For odd prime p, (-1|p) = 1 if p ≡ 1 (mod 4), and (-1|p) = -1 if p ≡ 3 (mod 4)
        self.assertEqual(custom_legendre(-1, 5), 1)  # 5 ≡ 1 (mod 4)
        self.assertEqual(custom_legendre(-1, 7), -1)  # 7 ≡ 3 (mod 4)
        self.assertEqual(custom_legendre(-1, 13), 1)  # 13 ≡ 1 (mod 4)
        self.assertEqual(custom_legendre(-1, 11), -1)  # 11 ≡ 3 (mod 4)

    def test_supplementary_laws(self):
        """Test supplementary laws of the Legendre symbol."""
        # For odd prime p:
        # (2|p) = 1 if p ≡ ±1 (mod 8), and (2|p) = -1 if p ≡ ±3 (mod 8)
        self.assertEqual(custom_legendre(2, 7), 1)  # 7 ≡ -1 (mod 8)
        self.assertEqual(custom_legendre(2, 17), 1)  # 17 ≡ 1 (mod 8)
        self.assertEqual(custom_legendre(2, 3), -1)  # 3 ≡ 3 (mod 8)
        self.assertEqual(custom_legendre(2, 11), -1)  # 11 ≡ 3 (mod 8)
        self.assertEqual(custom_legendre(2, 5), -1)  # 5 ≡ -3 (mod 8)
        self.assertEqual(custom_legendre(2, 13), -1)  # 13 ≡ 5 ≡ -3 (mod 8)

    def test_non_prime_moduli(self):
        """Test behavior with non-prime moduli (Jacobi symbol)."""
        # Note: The sympy.legendre_symbol function only works with prime moduli
        # We'll skip testing non-prime moduli as it's not supported by the function
        pass


if __name__ == "__main__":
    unittest.main()
