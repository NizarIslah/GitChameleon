import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_60 import get_slice


class TestGetSlice(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.series_numeric = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.series_string = pd.Series(["a", "b", "c", "d", "e"])
        self.series_mixed = pd.Series([1, "a", 3.14, None, True])
        self.series_empty = pd.Series([])

    def test_basic_slicing(self):
        # Test basic slicing with numeric indices
        result = get_slice(self.series_numeric, 2, 5)
        expected = pd.Series([3, 4, 5], index=[2, 3, 4])
        pd.testing.assert_series_equal(result, expected)

    def test_string_series(self):
        # Test slicing with a series of strings
        result = get_slice(self.series_string, 1, 3)
        expected = pd.Series(["b", "c"], index=[1, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_mixed_series(self):
        # Test slicing with a series of mixed types
        result = get_slice(self.series_mixed, 1, 4)
        expected = pd.Series(["a", 3.14, None], index=[1, 2, 3])
        pd.testing.assert_series_equal(result, expected)

    def test_full_slice(self):
        # Test slicing the entire series
        result = get_slice(self.series_numeric, 0, len(self.series_numeric))
        pd.testing.assert_series_equal(result, self.series_numeric)

    def test_empty_slice(self):
        # Test slicing with start == end (should return empty series)
        result = get_slice(self.series_numeric, 3, 3)
        expected = pd.Series([], dtype=self.series_numeric.dtype)
        pd.testing.assert_series_equal(result, expected)

    def test_empty_series(self):
        # Test slicing an empty series
        result = get_slice(self.series_empty, 0, 5)
        pd.testing.assert_series_equal(result, self.series_empty)

    def test_negative_indices(self):
        # Test slicing with negative indices
        result = get_slice(self.series_numeric, -5, -2)
        expected = pd.Series([6, 7, 8], index=[5, 6, 7])
        pd.testing.assert_series_equal(result, expected)

    def test_out_of_bounds(self):
        # Test slicing with indices out of bounds
        result = get_slice(self.series_numeric, 5, 20)
        expected = pd.Series([6, 7, 8, 9, 10], index=[5, 6, 7, 8, 9])
        pd.testing.assert_series_equal(result, expected)

    def test_reversed_indices(self):
        # Test slicing with start > end (should return empty series)
        result = get_slice(self.series_numeric, 8, 3)
        expected = pd.Series([], dtype=self.series_numeric.dtype)
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
