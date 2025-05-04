import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_113 import combine_pvalues


class TestSample113(unittest.TestCase):
    def test_combine_pvalues_basic(self):
        """Test the combine_pvalues function with a simple array of p-values."""
        # Create a sample array of p-values
        p_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Calculate the expected result using scipy directly
        expected_output = stats.combine_pvalues(p_values, "pearson")
        expected_result = (-expected_output[0], 1 - expected_output[1])

        # Get the actual result from our function
        actual_result = combine_pvalues(p_values)

        # Assert that the results are close (using almost equal for floating point comparison)
        self.assertAlmostEqual(actual_result[0], expected_result[0], places=10)
        self.assertAlmostEqual(actual_result[1], expected_result[1], places=10)

    def test_combine_pvalues_edge_cases(self):
        """Test the combine_pvalues function with edge cases."""
        # Test with an array of zeros (all p-values are 0)
        p_values_zeros = np.array([0.0, 0.0, 0.0])
        result_zeros = combine_pvalues(p_values_zeros)
        # The result should be a tuple of two floats
        self.assertIsInstance(result_zeros, tuple)
        self.assertEqual(len(result_zeros), 2)

        # Test with an array of ones (all p-values are 1)
        p_values_ones = np.array([1.0, 1.0, 1.0])
        result_ones = combine_pvalues(p_values_ones)
        # The result should be a tuple of two floats
        self.assertIsInstance(result_ones, tuple)
        self.assertEqual(len(result_ones), 2)

        # For p-values of 1, the combined p-value should also be 1
        self.assertAlmostEqual(result_ones[1], 1.0, places=10)

    def test_combine_pvalues_single_value(self):
        """Test the combine_pvalues function with a single p-value."""
        # Test with a single p-value
        p_value_single = np.array([0.05])
        result_single = combine_pvalues(p_value_single)

        # Calculate expected result
        expected_output = stats.combine_pvalues(p_value_single, "pearson")
        expected_result = (-expected_output[0], 1 - expected_output[1])

        # Assert that the results match
        self.assertAlmostEqual(result_single[0], expected_result[0], places=10)
        self.assertAlmostEqual(result_single[1], expected_result[1], places=10)


if __name__ == "__main__":
    unittest.main()
