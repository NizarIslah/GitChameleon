import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np
from PIL import Image, ImageChops

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_321 import imaging


class TestImaging(unittest.TestCase):
    def setUp(self):
        # Create test images
        # Create a 100x100 black image
        self.black_img = Image.new("RGB", (100, 100), color="black")

        # Create a 100x100 white image
        self.white_img = Image.new("RGB", (100, 100), color="white")

        # Create a 100x100 red image
        self.red_img = Image.new("RGB", (100, 100), color="red")

        # Create a 100x100 gradient image
        gradient = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient[:, i, :] = i * 255 // 99  # Normalize to 0-255
        self.gradient_img = Image.fromarray(gradient)

    def test_imaging_with_same_image(self):
        """Test that applying soft_light to the same image returns the same image."""
        result = imaging(self.black_img, self.black_img)
        # Check that result is an Image
        self.assertIsInstance(result, Image.Image)
        # For black + black, the result should still be black
        self.assertEqual(list(result.getdata())[0], (0, 0, 0))

    def test_imaging_with_red(self):
        """Test soft_light with a colored image."""
        result = imaging(self.red_img, self.gradient_img)

        # Verify result is an Image
        self.assertIsInstance(result, Image.Image)

        # Check dimensions are preserved
        self.assertEqual(result.size, self.red_img.size)

    def test_imaging_matches_imagechops(self):
        """Test that our function matches the behavior of ImageChops.soft_light."""
        # Apply our function
        result1 = imaging(self.gradient_img, self.red_img)

        # Apply ImageChops directly
        result2 = ImageChops.soft_light(self.gradient_img, self.red_img)

        # Convert images to arrays for comparison
        arr1 = np.array(result1)
        arr2 = np.array(result2)

        # They should be identical
        np.testing.assert_array_equal(arr1, arr2)


if __name__ == "__main__":
    unittest.main()
