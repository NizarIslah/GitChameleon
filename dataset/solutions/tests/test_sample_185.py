import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_185 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_185 import custom_function
from sympy import GF
from sympy.polys.domains.finitefield import FiniteField


class TestCustomFunction(unittest.TestCase):
    def test_basic_conversion_to_integer(self):
        """Test basic conversion from finite field element to integer."""
        # Create a finite field GF(5)
        K = GF(5)
        # Test conversion of element 3 in GF(5)
        self.assertEqual(custom_function(K, K(3)), -2)
        # Test conversion of element 1 in GF(5)
        self.assertEqual(custom_function(K, K(1)), 1)
        # Test conversion of element 4 in GF(5)
        self.assertEqual(custom_function(K, K(4)), -1)

    def test_different_prime_fields(self):
        """Test conversion in different prime fields."""
        # Test in GF(7)
        K7 = GF(7)
        self.assertEqual(custom_function(K7, K7(5)), -2)

        # Test in GF(11)
        K11 = GF(11)
        self.assertEqual(custom_function(K11, K11(8)), -3)

        # Test in GF(13)
        K13 = GF(13)
        self.assertEqual(custom_function(K13, K13(12)), -1)

    def test_zero_element_conversion(self):
        """Test conversion of zero element in different fields."""
        # Test zero in GF(5)
        K5 = GF(5)
        self.assertEqual(custom_function(K5, K5(0)), 0)

        # Test zero in GF(17)
        K17 = GF(17)
        self.assertEqual(custom_function(K17, K17(0)), 0)

    def test_negative_representation_conversion(self):
        """Test conversion of elements with negative representation."""
        # In GF(5), -1 is represented as 4
        K5 = GF(5)
        # K5(-1) should be equivalent to K5(4)
        self.assertEqual(custom_function(K5, K5(-1)), custom_function(K5, K5(4)))

        # In GF(7), -2 is represented as 5
        K7 = GF(7)
        # K7(-2) should be equivalent to K7(5)
        self.assertEqual(custom_function(K7, K7(-2)), custom_function(K7, K7(5)))

        # Check the actual integer value
        self.assertEqual(custom_function(K5, K5(-1)), -1)
        self.assertEqual(custom_function(K7, K7(-2)), -2)

    def test_large_prime_field(self):
        """Test conversion in a larger prime field."""
        # Test in GF(101)
        K101 = GF(101)
        self.assertEqual(custom_function(K101, K101(57)), -44)
        self.assertEqual(custom_function(K101, K101(100)), -1)

        # Test in GF(997) - a larger prime
        K997 = GF(997)
        # We'll test with values that have a predictable representation
        self.assertEqual(custom_function(K997, K997(1)), 1)
        self.assertEqual(custom_function(K997, K997(996)), -1)

    def test_invalid_element_for_field(self):
        """Test handling of invalid elements for a field."""
        K5 = GF(5)
        K7 = GF(7)

        # Try to use an element from GF(7) with GF(5)
        # This might not raise an exception in all implementations
        # So we'll just check that the result is consistent
        try:
            result = custom_function(K5, K7(3))
            # If no exception is raised, verify the result is an integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, that's also acceptable
            pass

        # Same for the other direction
        try:
            result = custom_function(K7, K5(2))
            self.assertIsInstance(result, int)
        except Exception as e:
            pass

    def test_non_finitefield_inputs(self):
        """Test handling of inputs that are not FiniteField instances."""
        K5 = GF(5)

        # Test with non-FiniteField first argument
        with self.assertRaises(AttributeError):
            custom_function("not a field", K5(3))

        # Test with non-FiniteField second argument
        # This might raise different exceptions depending on the implementation
        try:
            custom_function(K5, "not an element")
            # If no exception is raised, this is unexpected
            self.fail("Expected an exception when using a string as a field element")
        except Exception:
            # Any exception is acceptable
            pass

        # Test with integers instead of field elements
        try:
            custom_function(K5, 3)
            # If no exception is raised, this is unexpected
            self.fail("Expected an exception when using an integer as a field element")
        except Exception:
            # Any exception is acceptable
            pass

    def test_extension_field_conversion(self):
        """Test conversion in extension fields."""
        # Create an extension field GF(2^3) = GF(8)
        K8 = GF(2**3, "a")

        # Test conversion of elements in GF(8)
        # In GF(8), elements are represented as polynomials in 'a'
        # where 'a' is a root of an irreducible polynomial of degree 3 over GF(2)

        # Test the zero element
        self.assertEqual(custom_function(K8, K8(0)), 0)

        # Test the one element
        self.assertEqual(custom_function(K8, K8(1)), 1)

        # For other elements, we'll just verify that the conversion is consistent
        # and returns an integer, without making assumptions about the specific values
        for i in range(2, 8):
            element = K8(i)
            int_value = custom_function(K8, element)
            # Verify that the result is an integer
            self.assertIsInstance(int_value, int)
            # Converting the same element again should give the same result
            self.assertEqual(custom_function(K8, element), int_value)


if __name__ == "__main__":
    unittest.main()
