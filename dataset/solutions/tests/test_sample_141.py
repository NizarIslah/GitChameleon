import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_141 import apply_gaussian_filter


class TestGaussianFilter(unittest.TestCase):
    def test_apply_gaussian_filter_shape(self):
        """Test that the output shape matches the input shape."""
        # Create a 3D array (e.g., batch of images)
        input_array = np.random.rand(3, 10, 10)  # 3 images of 10x10
        sigma = 1.0

        result = apply_gaussian_filter(input_array, sigma)

        # Check that the shape is preserved
        self.assertEqual(input_array.shape, result.shape)

    def test_apply_gaussian_filter_values(self):
        """Test that the function applies filter correctly along axes 1 and 2."""
        # Create a simple 3D array
        input_array = np.ones((2, 5, 5))
        # Add a spike in the middle of each "image"
        input_array[0, 2, 2] = 10
        input_array[1, 2, 2] = 10

        sigma = 1.0

        # Apply our function
        result = apply_gaussian_filter(input_array, sigma)

        # Apply gaussian_filter directly with the same parameters for comparison
        expected = gaussian_filter(input_array, sigma=sigma, axes=[1, 2])

        # Check that results are the same
        np.testing.assert_allclose(result, expected)

    def test_different_sigma_values(self):
        """Test the function with different sigma values."""
        input_array = np.random.rand(2, 8, 8)

        # Test with small sigma
        small_sigma = 0.5
        result_small = apply_gaussian_filter(input_array, small_sigma)
        expected_small = gaussian_filter(input_array, sigma=small_sigma, axes=[1, 2])
        np.testing.assert_allclose(result_small, expected_small)

        # Test with large sigma
        large_sigma = 2.0
        result_large = apply_gaussian_filter(input_array, large_sigma)
        expected_large = gaussian_filter(input_array, sigma=large_sigma, axes=[1, 2])
        np.testing.assert_allclose(result_large, expected_large)

    def test_4d_array(self):
        """Test the function with a 4D array."""
        # Create a 4D array (e.g., batch of color images)
        input_array = np.random.rand(2, 3, 8, 8)  # 2 images, 3 channels, 8x8
        sigma = 1.0

        result = apply_gaussian_filter(input_array, sigma)

        # Check that the shape is preserved
        self.assertEqual(input_array.shape, result.shape)

        # Verify that the filter is applied correctly along axes 1 and 2
        expected = gaussian_filter(input_array, sigma=sigma, axes=[1, 2])
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
