import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_61 import get_slice


class TestGetSlice(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_get_slice_normal_range(self):
        # Test normal slicing
        result = get_slice(self.test_series, 2, 5)
        expected = pd.Series([3, 4, 5], index=[2, 3, 4])
        pd.testing.assert_series_equal(result, expected)

    def test_get_slice_start_to_end(self):
        # Test slicing from start to end
        result = get_slice(self.test_series, 0, 10)
        pd.testing.assert_series_equal(result, self.test_series)

    def test_get_slice_empty_result(self):
        # Test slicing that results in empty series
        result = get_slice(self.test_series, 5, 5)
        expected = pd.Series([], dtype=np.int64)
        pd.testing.assert_series_equal(result, expected)

    def test_get_slice_negative_indices(self):
        # Test with negative indices
        result = get_slice(self.test_series, -5, -2)
        expected = pd.Series([6, 7, 8], index=[5, 6, 7])
        pd.testing.assert_series_equal(result, expected)

    def test_get_slice_out_of_bounds(self):
        # Test with indices beyond series length
        result = get_slice(self.test_series, 8, 15)
        expected = pd.Series([9, 10], index=[8, 9])
        pd.testing.assert_series_equal(result, expected)

    def test_get_slice_with_non_numeric_index(self):
        # Test with non-numeric index
        series_with_labels = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
        result = get_slice(series_with_labels, 1, 4)
        expected = pd.Series([2, 3, 4], index=["b", "c", "d"])
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
