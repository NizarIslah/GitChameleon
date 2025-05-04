import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_317 import imaging


class TestImaging(unittest.TestCase):
    def setUp(self):
        # Create test images with known dimensions and values
        # Create a 3x3 black image
        self.black_img = Image.new("RGB", (3, 3), color=(0, 0, 0))

        # Create a 3x3 white image
        self.white_img = Image.new("RGB", (3, 3), color=(255, 255, 255))

        # Create a 3x3 gray image (value 127)
        self.gray_img = Image.new("RGB", (3, 3), color=(127, 127, 127))

        # Create a 3x3 image with different values for testing
        self.test_img = Image.new("RGB", (3, 3))
        test_data = np.zeros((3, 3, 3), dtype=np.uint8)
        # Set some pixels to values below 128 and some above
        test_data[0, 0] = [100, 150, 200]  # Mixed values
        test_data[0, 1] = [50, 50, 50]  # All below 128
        test_data[0, 2] = [200, 200, 200]  # All above 128
        test_data[1, 0] = [0, 127, 255]  # Min, mid, max
        test_data[1, 1] = [64, 128, 192]  # Below, at, above threshold
        test_data[1, 2] = [255, 0, 127]  # Max, min, mid
        test_data[2, 0] = [127, 127, 127]  # All at threshold
        test_data[2, 1] = [0, 0, 0]  # All black
        test_data[2, 2] = [255, 255, 255]  # All white
        self.test_array = test_data
        self.test_img = Image.fromarray(test_data)

        # Create a different sized image for testing shape mismatch
        self.different_size_img = Image.new("RGB", (4, 4), color=(100, 100, 100))

    def test_imaging_with_same_images(self):
        """Test overlay with identical images"""
        # Black overlay with black should remain black
        result = imaging(self.black_img, self.black_img)
        self.assertIsNotNone(result)
        result_array = np.array(result)
        self.assertTrue(np.all(result_array == 0))

        # White overlay with white should remain white
        result = imaging(self.white_img, self.white_img)
        self.assertIsNotNone(result)
        result_array = np.array(result)
        self.assertTrue(np.all(result_array == 255))

        # Gray overlay with gray
        result = imaging(self.gray_img, self.gray_img)
        self.assertIsNotNone(result)
        # The result should be calculated according to the overlay formula

    def test_imaging_black_white(self):
        """Test overlay between black and white images"""
        # Black overlay with white
        result = imaging(self.black_img, self.white_img)
        self.assertIsNotNone(result)
        result_array = np.array(result)
        # Black (0) * White (255) // 127 = 0
        self.assertTrue(np.all(result_array == 0))

        # White overlay with black
        result = imaging(self.white_img, self.black_img)
        self.assertIsNotNone(result)
        result_array = np.array(result)
        # For white pixels (>= 128): 255 - (((255-255) * (255-0)) // 127) = 255
        self.assertTrue(np.all(result_array == 255))

    def test_imaging_different_sizes(self):
        """Test overlay with different sized images"""
        # This should return None as per the create function
        result = imaging(self.black_img, self.different_size_img)
        self.assertIsNone(result)

    def test_imaging_mixed_values(self):
        """Test overlay with mixed pixel values"""
        # Test with gray image and test image
        result = imaging(self.gray_img, self.test_img)
        self.assertIsNotNone(result)

        # For each pixel in the result, verify it matches the expected calculation
        result_array = np.array(result)
        gray_value = 127

        for y in range(3):
            for x in range(3):
                for c in range(3):
                    test_val = self.test_array[y, x, c]
                    if gray_value < 128:
                        expected = np.clip((gray_value * test_val) // 127, 0, 255)
                    else:
                        expected = np.clip(
                            255 - (((255 - gray_value) * (255 - test_val)) // 127),
                            0,
                            255,
                        )
                    self.assertEqual(result_array[y, x, c], expected)


if __name__ == "__main__":
    unittest.main()
