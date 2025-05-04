import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_118 import compute_circular_variance


class TestCircularVariance(unittest.TestCase):
    def test_compute_circular_variance_zero(self):
        """Test circular variance of identical angles (should be zero)"""
        # Array with identical angles should have zero variance
        angles = np.array([0.5, 0.5, 0.5, 0.5]) * 2 * np.pi
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 0.0)

    def test_compute_circular_variance_uniform(self):
        """Test circular variance of uniformly distributed angles (should be 1)"""
        # Uniformly distributed angles around the circle should have variance close to 1
        angles = np.array([0, 0.25, 0.5, 0.75]) * 2 * np.pi
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 1.0)

    def test_compute_circular_variance_concentrated(self):
        """Test circular variance of concentrated angles"""
        # Angles concentrated in one direction
        angles = np.array([0.1, 0.11, 0.09, 0.095, 0.105]) * 2 * np.pi
        result = compute_circular_variance(angles)
        # Calculate expected result directly with scipy for comparison
        expected = stats.circvar(angles)
        self.assertAlmostEqual(result, expected)

    def test_compute_circular_variance_empty(self):
        """Test circular variance of empty array"""
        # Empty array should return nan
        angles = np.array([])
        result = compute_circular_variance(angles)
        self.assertTrue(np.isnan(result))

    def test_compute_circular_variance_single(self):
        """Test circular variance of single value"""
        # Single value should have zero variance
        angles = np.array([0.5]) * 2 * np.pi
        result = compute_circular_variance(angles)
        self.assertEqual(result, 0.0)

    def test_compute_circular_variance_opposite(self):
        """Test circular variance of opposite angles"""
        # Opposite angles should have high variance
        angles = np.array([0, np.pi])
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
