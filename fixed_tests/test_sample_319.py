import os
import sys
import unittest
import numpy as np
from PIL import Image

# Add the samples directory to the path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'samples'))
from sample_319 import imaging


class TestImaging(unittest.TestCase):
    def setUp(self):
        # Create test images
        self.img1_small = Image.new('RGB', (2, 2), color=(100, 150, 200))
        self.img2_small = Image.new('RGB', (2, 2), color=(50, 100, 150))
        self.img3_diff_size = Image.new('RGB', (3, 3), color=(100, 100, 100))

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
        self.expected_result[0, 0, 0] = (50 * 100) // 127
        self.expected_result[0, 0, 1] = 255 - ((255 - 150) * (255 - 200)) // 127
        self.expected_result[0, 0, 2] = (100 * 50) // 127
        self.expected_result[0, 1, 0] = 255 - ((255 - 100) * (255 - 150)) // 127
        self.expected_result[0, 1, 1] = (50 * 100) // 127
        self.expected_result[0, 1, 2] = 255 - ((255 - 150) * (255 - 200)) // 127
        self.expected_result[1, 0, 0] = 255 - ((255 - 150) * (255 - 200)) // 127
        self.expected_result[1, 0, 1] = (100 * 50) // 127
        self.expected_result[1, 0, 2] = (50 * 100) // 127
        self.expected_result[1, 1, 0] = (100 * 50) // 127
        self.expected_result[1, 1, 1] = 255 - ((255 - 200) * (255 - 150)) // 127
        self.expected_result[1, 1, 2] = 255 - ((255 - 150) * (255 - 200)) // 127

    def test_imaging_with_same_size_images(self):
        """Test that the imaging function works with same-sized images."""
        result = imaging(self.img1_small, self.img2_small)
        self.assertIsInstance(result, np.ndarray)  # Expecting a NumPy array
        self.assertEqual(result.shape, (2, 2, 3))  # Check shape instead of size

    def test_imaging_with_different_size_images(self):
        """Test that the imaging function returns None for different-sized images."""
        result = imaging(self.img1_small, self.img3_diff_size)
        self.assertIsNone(result)

    def test_hardlight_calculation(self):
        """Test that the hardlight calculation produces the expected results."""
        result = imaging(self.img_test1, self.img_test2)
        self.assertIsInstance(result, np.ndarray)  # Expecting a NumPy array
        
        # Compare with expected values
        np.testing.assert_array_equal(result, self.expected_result)

    def test_imaging_preserves_image_dimensions(self):
        """Test that the output image has the same dimensions as the input."""
        result = imaging(self.img_test1, self.img_test2)
        self.assertEqual(result.shape, (2, 2, 3))  # Check shape instead of size


if __name__ == '__main__':
    unittest.main()
