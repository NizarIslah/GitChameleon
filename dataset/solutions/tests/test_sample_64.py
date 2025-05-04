import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_64 import correct_type


class TestCorrectType(unittest.TestCase):
    def test_integer_index(self):
        """Test with an integer index."""
        index = pd.Index([1, 2, 3, 4, 5])
        self.assertEqual(correct_type(index), "int64")

    def test_float_index(self):
        """Test with a float index."""
        index = pd.Index([1.1, 2.2, 3.3, 4.4, 5.5])
        self.assertEqual(correct_type(index), "float64")

    def test_string_index(self):
        """Test with a string index."""
        index = pd.Index(["a", "b", "c", "d", "e"])
        self.assertEqual(correct_type(index), "object")

    def test_datetime_index(self):
        """Test with a datetime index."""
        index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03"])
        self.assertTrue(correct_type(index).startswith("datetime64"))

    def test_categorical_index(self):
        """Test with a categorical index."""
        index = pd.CategoricalIndex(["a", "b", "c", "a", "b"])
        self.assertEqual(correct_type(index), "category")

    def test_multi_index(self):
        """Test with a multi-index."""
        arrays = [["a", "a", "b", "b"], ["one", "two", "one", "two"]]
        index = pd.MultiIndex.from_arrays(arrays)
        self.assertEqual(correct_type(index), "object")


if __name__ == "__main__":
    unittest.main()
