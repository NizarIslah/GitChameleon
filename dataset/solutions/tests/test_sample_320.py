import unittest
import numpy as np
from PIL import Image, ImageChops
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_320 import imaging


class TestSample320(unittest.TestCase):
    def setUp(self):
        # Create two test images
        # Using numpy arrays to create images with specific pixel values
        self.array1 = np.array(
            [[[100, 150, 200], [120, 160, 210]], [[130, 170, 220], [140, 180, 230]]],
            dtype=np.uint8,
        )

        self.array2 = np.array(
            [[[50, 60, 70], [80, 90, 100]], [[110, 120, 130], [140, 150, 160]]],
            dtype=np.uint8,
        )

        self.img1 = Image.fromarray(self.array1)
        self.img2 = Image.fromarray(self.array2)

    def test_imaging_returns_image(self):
        """Test that the imaging function returns a PIL Image object."""
        result = imaging(self.img1, self.img2)
        self.assertIsInstance(result, Image.Image)

    def test_imaging_applies_overlay(self):
        """Test that the imaging function correctly applies the overlay operation."""
        result = imaging(self.img1, self.img2)

        # Manually apply the overlay operation to verify
        expected = ImageChops.overlay(self.img1, self.img2)

        # Convert images to arrays for comparison
        result_array = np.array(result)
        expected_array = np.array(expected)

        # Check if the arrays are equal
        np.testing.assert_array_equal(result_array, expected_array)

    def test_imaging_with_different_sized_images(self):
        """Test that the imaging function works with different sized images."""
        # Create a smaller image
        small_array = np.array([[[100, 150, 200]], [[130, 170, 220]]], dtype=np.uint8)
        small_img = Image.fromarray(small_array)

        # The overlay operation should crop to the smaller image
        result = imaging(self.img1, small_img)
        expected = ImageChops.overlay(self.img1, small_img)

        # Verify the result
        self.assertEqual(result.size, expected.size)

        # Convert images to arrays for comparison
        result_array = np.array(result)
        expected_array = np.array(expected)

        # Check if the arrays are equal
        np.testing.assert_array_equal(result_array, expected_array)


if __name__ == "__main__":
    unittest.main()
