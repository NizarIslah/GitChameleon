import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_117 import compute_circular_variance


class TestCircularVariance(unittest.TestCase):
    def test_zero_variance(self):
        """Test when all angles are the same, variance should be 0."""
        angles = np.array([0, 0, 0, 0])
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 0.0)

        # Test with different angle value
        angles = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 0.0)

    def test_maximum_variance(self):
        """Test when angles are uniformly distributed, variance should be close to 1."""
        # Angles evenly distributed around the circle
        angles = np.array([0, np.pi, 2 * np.pi / 3, 4 * np.pi / 3])
        result = compute_circular_variance(angles)
        # The expected value is 0.75 for these four angles
        self.assertAlmostEqual(result, 0.75, places=10)

        # Another example with uniform distribution
        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_intermediate_variance(self):
        """Test with angles that should give intermediate variance values."""
        # Angles with some dispersion but not maximum
        angles = np.array([0, np.pi / 6, -np.pi / 6])
        result = compute_circular_variance(angles)
        # The expected value is approximately 1 - |mean of exp(i*angles)|
        expected = 1 - np.abs(np.mean(np.exp(1j * angles)))
        self.assertAlmostEqual(result, expected)

        # Another example
        angles = np.array([0, np.pi / 4, np.pi / 2])
        result = compute_circular_variance(angles)
        expected = 1 - np.abs(np.mean(np.exp(1j * angles)))
        self.assertAlmostEqual(result, expected)

    def test_empty_array(self):
        """Test with an empty array, should return NaN."""
        angles = np.array([])
        result = compute_circular_variance(angles)
        self.assertTrue(np.isnan(result))

    def test_single_value(self):
        """Test with a single value, variance should be 0."""
        angles = np.array([np.pi / 3])
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 0.0)

    def test_opposite_angles(self):
        """Test with opposite angles, variance should be 1."""
        angles = np.array([0, np.pi])
        result = compute_circular_variance(angles)
        self.assertAlmostEqual(result, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
