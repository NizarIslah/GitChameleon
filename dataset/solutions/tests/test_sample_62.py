import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_62 import correct_type


class TestCorrectType(unittest.TestCase):
    def test_correct_type_returns_int64(self):
        """Test that correct_type always returns 'int64' regardless of input."""
        # Test with integer index
        int_index = pd.Index([1, 2, 3, 4, 5])
        self.assertEqual(correct_type(int_index), "int64")

        # Test with string index
        str_index = pd.Index(["a", "b", "c"])
        self.assertEqual(correct_type(str_index), "int64")

        # Test with float index
        float_index = pd.Index([1.1, 2.2, 3.3])
        self.assertEqual(correct_type(float_index), "int64")

        # Test with datetime index
        date_index = pd.date_range("2023-01-01", periods=5)
        self.assertEqual(correct_type(date_index), "int64")

        # Test with empty index
        empty_index = pd.Index([])
        self.assertEqual(correct_type(empty_index), "int64")


if __name__ == "__main__":
    unittest.main()
