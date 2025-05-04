import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_184 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_184 import custom_function
from sympy import divisors


class TestCustomFunction(unittest.TestCase):
    def test_basic_divisor_sigma_functionality(self):
        """Test basic functionality of divisor_sigma through custom_function."""
        # divisor_sigma(8, 1) = 1 + 2 + 4 + 8 = 15
        self.assertEqual(custom_function(8, 1), 15)
        # divisor_sigma(12, 1) = 1 + 2 + 3 + 4 + 6 + 12 = 28
        self.assertEqual(custom_function(12, 1), 28)
        # divisor_sigma(10, 1) = 1 + 2 + 5 + 10 = 18
        self.assertEqual(custom_function(10, 1), 18)

    def test_different_k_values(self):
        """Test with different k values."""
        # divisor_sigma(6, 0) = count of divisors = 4 (1, 2, 3, 6)
        self.assertEqual(custom_function(6, 0), 4)
        # divisor_sigma(6, 1) = 1 + 2 + 3 + 6 = 12
        self.assertEqual(custom_function(6, 1), 12)
        # divisor_sigma(6, 2) = 1^2 + 2^2 + 3^2 + 6^2 = 1 + 4 + 9 + 36 = 50
        self.assertEqual(custom_function(6, 2), 50)
        # divisor_sigma(6, 3) = 1^3 + 2^3 + 3^3 + 6^3 = 1 + 8 + 27 + 216 = 252
        self.assertEqual(custom_function(6, 3), 252)

    def test_large_numbers(self):
        """Test with larger numbers."""
        # divisor_sigma(100, 1) = sum of all divisors of 100
        self.assertEqual(custom_function(100, 1), 217)
        # divisor_sigma(1000, 1) = sum of all divisors of 1000
        self.assertEqual(custom_function(1000, 1), 2340)

    def test_edge_case_n_equals_one(self):
        """Test with n=1 for different k values."""
        # divisor_sigma(1, 0) = 1 (count of divisors)
        self.assertEqual(custom_function(1, 0), 1)
        # divisor_sigma(1, 1) = 1 (sum of divisors)
        self.assertEqual(custom_function(1, 1), 1)
        # divisor_sigma(1, 2) = 1^2 = 1
        self.assertEqual(custom_function(1, 2), 1)

    def test_negative_n_values(self):
        """Test with negative n values, which should raise exceptions."""
        with self.assertRaises(ValueError):
            custom_function(-1, 1)
        with self.assertRaises(ValueError):
            custom_function(-10, 2)

    def test_negative_k_values(self):
        """Test with negative k values, which should raise exceptions."""
        # SymPy's divisor_sigma doesn't support negative k values
        with self.assertRaises(ValueError):
            custom_function(6, -1)
        with self.assertRaises(ValueError):
            custom_function(12, -2)

    def test_non_integer_inputs(self):
        """Test with non-integer inputs."""
        # For string inputs, SymPy returns symbolic expressions
        result1 = custom_function("string", 1)
        self.assertEqual(str(result1), "divisor_sigma(string, 1)")

        result2 = custom_function(10, "string")
        self.assertEqual(str(result2), "divisor_sigma(10, string)")

        # For float inputs, SymPy treats them as symbolic if they're not exact integers
        result3 = custom_function(10.0, 1)
        self.assertEqual(str(result3), "divisor_sigma(10.0, 1)")

    def test_compare_with_manual_calculation(self):
        """Test by comparing with manual calculation using divisors function."""
        test_cases = [(6, 1), (8, 2), (15, 3), (20, 0)]

        for n, k in test_cases:
            with self.subTest(n=n, k=k):
                # Calculate manually using divisors function
                manual_result = sum(d**k for d in divisors(n))
                # Compare with custom_function result
                self.assertEqual(custom_function(n, k), manual_result)


if __name__ == "__main__":
    unittest.main()
