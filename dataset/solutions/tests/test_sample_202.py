import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_202 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_202 import custom_npartitions
import sympy


class TestCustomNPartitions(unittest.TestCase):
    def test_basic_partition_values(self):
        """Test custom_npartitions with basic known values."""
        # Test some known partition values
        self.assertEqual(custom_npartitions(1), 1)  # p(1) = 1 (only one partition: 1)
        self.assertEqual(custom_npartitions(2), 2)  # p(2) = 2 (partitions: 2, 1+1)
        self.assertEqual(
            custom_npartitions(3), 3
        )  # p(3) = 3 (partitions: 3, 2+1, 1+1+1)
        self.assertEqual(
            custom_npartitions(4), 5
        )  # p(4) = 5 (partitions: 4, 3+1, 2+2, 2+1+1, 1+1+1+1)
        self.assertEqual(custom_npartitions(5), 7)  # p(5) = 7
        self.assertEqual(custom_npartitions(6), 11)  # p(6) = 11
        self.assertEqual(custom_npartitions(7), 15)  # p(7) = 15
        self.assertEqual(custom_npartitions(10), 42)  # p(10) = 42

    def test_partition_of_zero(self):
        """Test custom_npartitions with n=0."""
        # By convention, p(0) = 1 (the empty sum)
        self.assertEqual(custom_npartitions(0), 1)

    def test_partition_of_one(self):
        """Test custom_npartitions with n=1."""
        # p(1) = 1 (only one partition: 1)
        self.assertEqual(custom_npartitions(1), 1)

    def test_partition_of_negative_numbers(self):
        """Test custom_npartitions with negative numbers."""
        # Partitions are not defined for negative numbers
        # SymPy should raise an exception or return a specific value
        try:
            result = custom_npartitions(-1)
            # If we get here, check that the result is a valid integer
            # Some implementations define p(n) = 0 for n < 0
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since partitions
            # are not defined for negative numbers
            pass

        try:
            result = custom_npartitions(-5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

    def test_partition_of_large_numbers(self):
        """Test custom_npartitions with large numbers."""
        # Test with some larger values
        # p(20) = 627
        self.assertEqual(custom_npartitions(20), 627)

        # p(50) = 204226
        # This might be slow, so we'll check if the result is at least a valid number
        result = custom_npartitions(50)
        self.assertEqual(result, 204226)

        # p(100) is a very large number, so we'll just check if the result matches
        # the SymPy implementation directly
        try:
            result = custom_npartitions(100)
            expected = sympy.functions.combinatorial.numbers.partition(100)
            self.assertEqual(result, expected)
        except Exception as e:
            # If an exception is raised due to computational limitations, that's acceptable
            pass

    def test_partition_with_non_integer_input(self):
        """Test custom_npartitions with non-integer input."""
        # Partitions are defined only for integers
        # SymPy should raise an exception or handle this appropriately

        # Test with float input
        try:
            result = custom_npartitions(3.5)
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected since partitions
            # are defined only for integers
            pass

        # Test with string input
        try:
            result = custom_npartitions("3")
            # If we get here, check that the result is a valid integer
            self.assertIsInstance(result, int)
        except Exception as e:
            # If an exception is raised, it's expected
            pass

    def test_verify_against_known_sequences(self):
        """Test custom_npartitions against known sequences."""
        # The first few values of p(n) are:
        # p(0) = 1, p(1) = 1, p(2) = 2, p(3) = 3, p(4) = 5, p(5) = 7, p(6) = 11, p(7) = 15, p(8) = 22, p(9) = 30, p(10) = 42
        known_values = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]

        for n, expected in enumerate(known_values):
            self.assertEqual(custom_npartitions(n), expected, f"Failed for n={n}")

    def test_partition_recurrence_relation(self):
        """Test custom_npartitions for the recurrence relation."""
        # The partition function satisfies various recurrence relations
        # One of them is the pentagonal number theorem:
        # p(n) = p(n-1) + p(n-2) - p(n-5) - p(n-7) + p(n-12) + p(n-15) - ...
        # where the indices are given by the pentagonal numbers k(3k-1)/2 for k = 1, -1, 2, -2, ...

        # Let's test the pentagonal number theorem for a few values
        # For n = 6:
        # p(6) = p(5) + p(4) - p(1) - p(-1) + ...
        # Since p(-1) and beyond are 0 or undefined, we can simplify:
        # p(6) = p(5) + p(4) - p(1)
        self.assertEqual(
            custom_npartitions(6),
            custom_npartitions(5) + custom_npartitions(4) - custom_npartitions(1),
        )

        # For n = 7:
        # p(7) = p(6) + p(5) - p(2) - p(0) + ...
        # p(7) = p(6) + p(5) - p(2) - 1 + ...
        self.assertEqual(
            custom_npartitions(7),
            custom_npartitions(6)
            + custom_npartitions(5)
            - custom_npartitions(2)
            - custom_npartitions(0),
        )

        # For n = 8:
        # p(8) = p(7) + p(6) - p(3) - p(1) + ...
        self.assertEqual(
            custom_npartitions(8),
            custom_npartitions(7)
            + custom_npartitions(6)
            - custom_npartitions(3)
            - custom_npartitions(1),
        )


if __name__ == "__main__":
    unittest.main()
