import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_114 import combine_pvalues


class TestCombinePValues(unittest.TestCase):
    def test_combine_pvalues_basic(self):
        # Test with a simple array of p-values
        p_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = combine_pvalues(p_values)

        # Check that the result is a tuple of two floats
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

        # Verify the result matches scipy's implementation
        expected = stats.combine_pvalues(p_values, "pearson")
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])

    def test_combine_pvalues_edge_cases(self):
        # Test with extreme p-values
        p_values = np.array([0.001, 0.999])
        result = combine_pvalues(p_values)
        expected = stats.combine_pvalues(p_values, "pearson")
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])

        # Test with a single p-value
        p_values = np.array([0.05])
        result = combine_pvalues(p_values)
        expected = stats.combine_pvalues(p_values, "pearson")
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])

    def test_combine_pvalues_zeros_and_ones(self):
        # Test with zeros and ones (boundary values)
        p_values = np.array([0.0, 1.0, 0.5])
        result = combine_pvalues(p_values)
        expected = stats.combine_pvalues(p_values, "pearson")
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])


if __name__ == "__main__":
    unittest.main()
