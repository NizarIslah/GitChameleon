# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_128 import apply_gaussian_filter1d


class TestGaussianFilter1D(unittest.TestCase):
    def test_apply_gaussian_filter1d_basic(self):
        """Test that the function applies the filter correctly."""
        # Create a simple input array
        x = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        radius = 3
        sigma = 1.0

        # Apply our function
        result = apply_gaussian_filter1d(x, radius, sigma)

        # Apply the filter directly with equivalent parameters for comparison
        expected = gaussian_filter1d(x, sigma=sigma, truncate=radius / sigma)

        # Check that results match
        np.testing.assert_allclose(result, expected)

    def test_apply_gaussian_filter1d_random_data(self):
        """Test with random data."""
        # Create random input
        np.random.seed(42)  # For reproducibility
        x = np.random.rand(100)
        radius = 2
        sigma = 0.8

        # Apply our function
        result = apply_gaussian_filter1d(x, radius, sigma)

        # Apply the filter directly with equivalent parameters
        expected = gaussian_filter1d(x, sigma=sigma, truncate=radius / sigma)

        # Check that results match
        np.testing.assert_allclose(result, expected)

    def test_apply_gaussian_filter1d_edge_cases(self):
        """Test edge cases."""
        # Test with very small sigma
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        radius = 2
        sigma = 0.1

        result = apply_gaussian_filter1d(x, radius, sigma)
        expected = gaussian_filter1d(x, sigma=sigma, truncate=radius / sigma)

        np.testing.assert_allclose(result, expected)

        # Test with large radius
        radius = 10
        sigma = 1.0

        result = apply_gaussian_filter1d(x, radius, sigma)
        expected = gaussian_filter1d(x, sigma=sigma, truncate=radius / sigma)

        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
