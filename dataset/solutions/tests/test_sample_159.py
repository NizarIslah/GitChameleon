import os
import sys
import unittest

import numpy as np
from scipy.stats import hmean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_159 import count_unique_hmean


class TestCountUniqueHmean(unittest.TestCase):
    def test_basic_functionality(self):
        # Test with a simple 2D array
        data = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        result = count_unique_hmean(data)

        # Calculate expected result manually
        expected_hmeans = hmean(data, axis=1)
        expected_unique_count = np.unique(expected_hmeans, equal_nan=False).shape[0]

        self.assertEqual(result, expected_unique_count)

    def test_identical_rows(self):
        # Test with rows that have the same harmonic mean
        data = np.array(
            [[1, 2, 3], [2, 4, 6]]
        )  # These rows are proportional, so hmean should be proportional
        result = count_unique_hmean(data)

        # Calculate expected result manually
        expected_hmeans = hmean(data, axis=1)
        expected_unique_count = np.unique(expected_hmeans, equal_nan=False).shape[0]

        self.assertEqual(result, expected_unique_count)

    def test_single_row(self):
        # Test with a single row
        data = np.array([[1, 2, 3]])
        result = count_unique_hmean(data)
        self.assertEqual(result, 1)

    def test_empty_array(self):
        # Test with an empty array (should handle this case)
        data = np.array([])
        with self.assertRaises(ValueError):
            count_unique_hmean(data)

    def test_with_negative_values(self):
        # Test with negative values (hmean should raise an error for negative values)
        data = np.array([[1, 2, -3], [3, 4, 5]])
        with self.assertRaises(ValueError):
            count_unique_hmean(data)

    def test_large_array(self):
        # Test with a larger array
        data = np.random.randint(1, 100, size=(100, 5))
        result = count_unique_hmean(data)

        # Calculate expected result manually
        expected_hmeans = hmean(data, axis=1)
        expected_unique_count = np.unique(expected_hmeans, equal_nan=False).shape[0]

        self.assertEqual(result, expected_unique_count)


if __name__ == "__main__":
    unittest.main()
