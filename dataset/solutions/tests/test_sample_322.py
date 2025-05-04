import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np
from PIL import Image, ImageChops

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_322 import imaging


class TestImaging(unittest.TestCase):
    def setUp(self):
        # Create two test images
        # First image: a simple 100x100 red image
        self.img1 = Image.new("RGB", (100, 100), color="red")

        # Second image: a simple 100x100 blue image
        self.img2 = Image.new("RGB", (100, 100), color="blue")

        # Create a gradient image for more complex testing
        gradient = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient[:, i, 0] = i * 255 // 100  # Red gradient
            gradient[:, i, 1] = (100 - i) * 255 // 100  # Green gradient
        self.gradient_img = Image.fromarray(gradient)

    def test_imaging_returns_image(self):
        """Test that the imaging function returns a PIL Image object."""
        result = imaging(self.img1, self.img2)
        self.assertIsInstance(result, Image.Image)

    def test_imaging_dimensions(self):
        """Test that the output image has the same dimensions as the inputs."""
        result = imaging(self.img1, self.img2)
        self.assertEqual(result.size, self.img1.size)
        self.assertEqual(result.size, self.img2.size)

    def test_imaging_mode(self):
        """Test that the output image has the same mode as the inputs."""
        result = imaging(self.img1, self.img2)
        self.assertEqual(result.mode, self.img1.mode)

    def test_imaging_hard_light_effect(self):
        """Test that the imaging function correctly applies the hard_light effect."""
        # We'll compare our function's output with direct call to ImageChops.hard_light
        expected = ImageChops.hard_light(self.img1, self.img2)
        result = imaging(self.img1, self.img2)

        # Convert images to numpy arrays for comparison
        expected_array = np.array(expected)
        result_array = np.array(result)

        # Check if the arrays are identical
        np.testing.assert_array_equal(result_array, expected_array)

    def test_with_gradient_image(self):
        """Test the function with a more complex gradient image."""
        result = imaging(self.gradient_img, self.img2)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.gradient_img.size)

        # Verify it matches the expected output from ImageChops.hard_light
        expected = ImageChops.hard_light(self.gradient_img, self.img2)
        expected_array = np.array(expected)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, expected_array)


if __name__ == "__main__":
    unittest.main()
