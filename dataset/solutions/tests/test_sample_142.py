import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_142 import apply_gaussian_filter


class TestGaussianFilter(unittest.TestCase):
    def test_apply_gaussian_filter_1d(self):
        """Test the function with a 1D array."""
        # Create a test array
        test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = 1.0

        # Apply our function
        result = apply_gaussian_filter(test_array.reshape(5, 1), sigma)

        # Apply gaussian_filter directly for comparison
        expected = np.zeros((5, 1))
        for i in range(5):
            expected[i] = gaussian_filter(test_array.reshape(5, 1)[i], sigma=sigma)

        # Check if results match
        np.testing.assert_allclose(result, expected)

    def test_apply_gaussian_filter_2d(self):
        """Test the function with a 2D array."""
        # Create a test array
        test_array = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        sigma = 0.5

        # Apply our function
        result = apply_gaussian_filter(test_array, sigma)

        # Apply gaussian_filter directly for comparison
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = gaussian_filter(test_array[i], sigma=sigma)

        # Check if results match
        np.testing.assert_allclose(result, expected)

    def test_apply_gaussian_filter_3d(self):
        """Test the function with a 3D array."""
        # Create a test array
        test_array = np.random.rand(3, 4, 5)
        sigma = 1.2

        # Apply our function
        result = apply_gaussian_filter(test_array, sigma)

        # Apply gaussian_filter directly for comparison
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = gaussian_filter(test_array[i], sigma=sigma)

        # Check if results match
        np.testing.assert_allclose(result, expected)

    def test_zero_sigma(self):
        """Test with sigma=0, which should return the original array."""
        test_array = np.random.rand(2, 3, 3)
        sigma = 0.0

        result = apply_gaussian_filter(test_array, sigma)

        # With sigma=0, the result should be very close to the original
        expected = np.zeros_like(test_array)
        for i in range(test_array.shape[0]):
            expected[i] = gaussian_filter(test_array[i], sigma=sigma)

        np.testing.assert_allclose(result, expected)

    def test_empty_array(self):
        """Test with an empty array."""
        test_array = np.array([[], []])
        test_array = test_array.reshape(2, 0)
        sigma = 1.0

        result = apply_gaussian_filter(test_array, sigma)

        # Expected shape should match input shape
        self.assertEqual(result.shape, test_array.shape)


if __name__ == "__main__":
    unittest.main()
