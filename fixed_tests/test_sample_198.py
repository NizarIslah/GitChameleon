import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_198 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_198 import custom_is_prime


class TestCustomIsPrime(unittest.TestCase):
    def test_with_prime_numbers(self):
        """Test custom_is_prime with prime numbers."""
        prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for num in prime_numbers:
            with self.subTest(num=num):
                self.assertTrue(custom_is_prime(num))
    
    def test_with_non_prime_numbers(self):
        """Test custom_is_prime with non-prime numbers."""
        non_prime_numbers = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21]
        for num in non_prime_numbers:
            with self.subTest(num=num):
                self.assertFalse(custom_is_prime(num))
    
    def test_with_zero_and_one(self):
        """Test custom_is_prime with zero and one."""
        self.assertFalse(custom_is_prime(0))
        self.assertFalse(custom_is_prime(1))
    
    def test_with_negative_numbers(self):
        """Test custom_is_prime with negative numbers."""
        negative_numbers = [-1, -2, -3, -4, -5]
        for num in negative_numbers:
            with self.subTest(num=num):
                self.assertFalse(custom_is_prime(num))
    
    def test_with_large_numbers(self):
        """Test custom_is_prime with large numbers."""
        self.assertTrue(custom_is_prime(104729))
        self.assertTrue(custom_is_prime(15485863))
        self.assertFalse(custom_is_prime(104730))  # 104729 + 1
        self.assertFalse(custom_is_prime(15485864))  # 15485863 + 1
    
    def test_with_non_integer_input(self):
        """Test custom_is_prime with non-integer input."""
        non_integer_inputs = [3.0, 4.0, 3.5]
        for input_value in non_integer_inputs:
            with self.subTest(input_value=input_value):
                with self.assertRaises((ValueError, TypeError)):
                    custom_is_prime(input_value)
    
    def test_with_edge_cases(self):
        """Test custom_is_prime with edge cases."""
        self.assertTrue(custom_is_prime(2))
        self.assertTrue(custom_is_prime(31))
        self.assertTrue(custom_is_prime(127))
        self.assertTrue(custom_is_prime(17))
        self.assertTrue(custom_is_prime(257))
    
    def test_return_type(self):
        """Test that the return type is a boolean."""
        result = custom_is_prime(3)
        self.assertIsInstance(result, bool)
        
        result = custom_is_prime(4)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
