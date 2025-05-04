import os
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_127 import apply_gaussian_filter1d


class TestGaussianFilter1D(unittest.TestCase):
    def test_apply_gaussian_filter1d_zeros(self):
        """Test with an array of zeros."""
        x = np.zeros(10)
        radius = 3
        sigma = 1.5

        # Expected result
        expected = gaussian_filter1d(x, sigma=sigma)

        # Result from our function
        result = apply_gaussian_filter1d(x, radius=radius, sigma=sigma)

        # Check that the results are the same
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
