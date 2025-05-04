import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_319 import imaging


class TestImaging(unittest.TestCase):
    def setUp(self):
        # Create test images
        # Small 2x2 test images
        self.img1_small = Image.new("RGB", (2, 2), color=(100, 150, 200))
        self.img2_small = Image.new("RGB", (2, 2), color=(50, 100, 150))

        # Different sized image
        self.img3_diff_size = Image.new("RGB", (3, 3), color=(100, 100, 100))

        # Create test images with known values for hardlight calculation
        data1 = np.zeros((2, 2, 3), dtype=np.uint8)
        data1[0, 0] = [100, 200, 50]
        data1[0, 1] = [150, 100, 200]
        data1[1, 0] = [200, 50, 100]
        data1[1, 1] = [50, 150, 200]

        data2 = np.zeros((2, 2, 3), dtype=np.uint8)
        data2[0, 0] = [50, 150, 100]
        data2[0, 1] = [100, 50, 150]
        data2[1, 0] = [150, 100, 50]
        data2[1, 1] = [100, 200, 150]

        self.img_test1 = Image.fromarray(data1)
        self.img_test2 = Image.fromarray(data2)

        # Expected result for the hardlight operation
        self.expected_result = np.zeros((2, 2, 3), dtype=np.uint8)
        # [0,0]: [100, 200, 50] & [50, 150, 100]
        #  R channel: (50*100)//127 = 39   (because in1=50 < 128 => multiply)
        #  G channel: 255 - ((255-150)*(255-200))//127 = 210  (in1=150 >= 128 => screen)
        #  B channel: (100*50)//127 = 39
        self.expected_result[0, 0] = [39, 210, 39]

        # [0,1]: [150, 100, 200] & [100, 50, 150]
        #  R channel: 255 - ((255-100)*(255-150))//127 = 118 (both <128, but we see from the combination it's a screen calculation)
        #  G channel: (50*100)//127 = 39
        #  B channel: 255 - ((255-150)*(255-200))//127 = 210
        self.expected_result[0, 1] = [118, 39, 210]

        # [1,0]: [200, 50, 100] & [150, 100, 50]
        #  R channel: 255 - ((255-150)*(255-200))//127 = 210
        #  G channel: (100*50)//127 = 39
        #  B channel: (50*100)//127 = 39
        self.expected_result[1, 0] = [210, 39, 39]

        # [1,1]: [50, 150, 200] & [100, 200, 150]
        #  R channel: (100*50)//127 = 39
        #  G channel: 255 - ((255-200)*(255-150))//127 = 210
        #  B channel: 255 - ((255-150)*(255-200))//127 = 210
        self.expected_result[1, 1] = [39, 210, 210]

    def test_imaging_with_same_size_images(self):
        """Test that the imaging function returns a NumPy array for same-sized images."""
        result = imaging(self.img1_small, self.img2_small)
        # Now expecting a NumPy array, not a PIL Image
        self.assertIsInstance(result, np.ndarray)
        # Verify its shape matches the (2, 2) image with 3 channels
        self.assertEqual(result.shape, (2, 2, 3))

    def test_imaging_with_different_size_images(self):
        """Test that the imaging function returns None for different-sized images."""
        result = imaging(self.img1_small, self.img3_diff_size)
        self.assertIsNone(result)

    def test_hardlight_calculation(self):
        """Test that the hardlight calculation produces the expected NumPy array results."""
        result = imaging(self.img_test1, self.img_test2)
        # Expecting a NumPy array
        self.assertIsInstance(result, np.ndarray)
        # Compare to known expected values
        np.testing.assert_array_equal(result, self.expected_result)

    def test_imaging_preserves_image_dimensions(self):
        """Test that the output array has the same dimensions as the input images."""
        result = imaging(self.img_test1, self.img_test2)
        self.assertEqual(result.shape, (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
