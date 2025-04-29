import os
# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'samples'))
from sample_319 import imaging


class TestImaging(unittest.TestCase):
    def setUp(self):
        # Create test images
        # Small 2x2 test images
        self.img1_small = Image.new('RGB', (2, 2), color=(100, 150, 200))
        self.img2_small = Image.new('RGB', (2, 2), color=(50, 100, 150))
        
        # Different sized image
        self.img3_diff_size = Image.new('RGB', (3, 3), color=(100, 100, 100))
        
        # Create test images with known values for hardlight calculation
        # Case 1: in1 < 128 (multiply blend)
        # For in1=100, in2=50: (100*50)//127 = 39
        # Case 2: in1 >= 128 (screen blend)
        # For in1=200, in2=150: 255 - ((255-200)*(255-150))//127 = 255 - (55*105)//127 = 255 - 45 = 210
        data1 = np.zeros((2, 2, 3), dtype=np.uint8)
        data1[0, 0] = [100, 200, 50]  # Test both cases in different channels
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
        # Calculate expected values manually
        # [0,0]: [100,200,50] and [50,150,100]
        self.expected_result[0, 0, 0] = (50 * 100) // 127  # in1=50, in2=100, < 128
        self.expected_result[0, 0, 1] = 255 - ((255 - 150) * (255 - 200)) // 127  # in1=150, in2=200, >= 128
        self.expected_result[0, 0, 2] = (100 * 50) // 127  # in1=100, in2=50, < 128
        
        # [0,1]: [150,100,200] and [100,50,150]
        self.expected_result[0, 1, 0] = 255 - ((255 - 100) * (255 - 150)) // 127  # in1=100, in2=150, < 128
        self.expected_result[0, 1, 1] = (50 * 100) // 127  # in1=50, in2=100, < 128
        self.expected_result[0, 1, 2] = 255 - ((255 - 150) * (255 - 200)) // 127  # in1=150, in2=200, >= 128
        
        # [1,0]: [200,50,100] and [150,100,50]
        self.expected_result[1, 0, 0] = 255 - ((255 - 150) * (255 - 200)) // 127  # in1=150, in2=200, >= 128
        self.expected_result[1, 0, 1] = (100 * 50) // 127  # in1=100, in2=50, < 128
        self.expected_result[1, 0, 2] = (50 * 100) // 127  # in1=50, in2=100, < 128
        
        # [1,1]: [50,150,200] and [100,200,150]
        self.expected_result[1, 1, 0] = (100 * 50) // 127  # in1=100, in2=50, < 128
        self.expected_result[1, 1, 1] = 255 - ((255 - 200) * (255 - 150)) // 127  # in1=200, in2=150, >= 128
        self.expected_result[1, 1, 2] = 255 - ((255 - 150) * (255 - 200)) // 127  # in1=150, in2=200, >= 128

    def test_imaging_with_same_size_images(self):
        """Test that the imaging function works with same-sized images."""
        result = imaging(self.img1_small, self.img2_small)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2, 3))

    def test_imaging_with_different_size_images(self):
        """Test that the imaging function returns None for different-sized images."""
        result = imaging(self.img1_small, self.img3_diff_size)
        self.assertIsNone(result)

    def test_hardlight_calculation(self):
        """Test that the hardlight calculation produces the expected results."""
        result = imaging(self.img_test1, self.img_test2)
        self.assertIsInstance(result, np.ndarray)
        # Compare with expected values
        np.testing.assert_array_equal(result, self.expected_result)

    def test_imaging_preserves_image_dimensions(self):
        """Test that the output image has the same dimensions as the input."""
        result = imaging(self.img_test1, self.img_test2)
        self.assertEqual(result.shape, (2, 2, 3))


if __name__ == '__main__':
    unittest.main()
