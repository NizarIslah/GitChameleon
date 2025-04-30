import os
# Add the parent directory to the path so we can import the sample
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_318 import imaging


class TestSample318(unittest.TestCase):
    def setUp(self):
        # Create test images of the same size
        self.img1 = Image.new('RGB', (10, 10), color=(100, 150, 200))
        self.img2 = Image.new('RGB', (10, 10), color=(50, 100, 150))
        
        # Create test images of different sizes
        self.img3 = Image.new('RGB', (5, 5), color=(100, 150, 200))
        
        # Create a black and white image for predictable results
        self.black = Image.new('RGB', (10, 10), color=(0, 0, 0))
        self.white = Image.new('RGB', (10, 10), color=(255, 255, 255))

    def test_same_size_images(self):
        """Test that the function works with same-sized images."""
        result = imaging(self.img1, self.img2)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img1.size)

    def test_different_size_images(self):
        """Test that the function returns None for different-sized images."""
        # The inner function imaging_softlight should return None, but the outer function
        # will convert this to a PIL Image, so we need to check if the result is valid
        result = imaging(self.img1, self.img3)
        self.assertIsNone(result)

    def test_black_on_white(self):
        """Test softlight blending with black on white."""
        result = imaging(self.white, self.black)
        # Convert result back to numpy for easier assertion
        result_array = np.array(result)
        # White softlight blended with black should result in white
        # The formula: ((255-255)*(255*0))/65536 + (255*(255-((255-255)*(255-0))/255))/255 = 255
        self.assertTrue(np.all(result_array == 255))

    def test_white_on_black(self):
        """Test softlight blending with white on black."""
        result = imaging(self.black, self.white)
        # Convert result back to numpy for easier assertion
        result_array = np.array(result)
        # Black softlight blended with white should result in black
        # The formula: ((255-0)*(0*255))/65536 + (0*(255-((255-0)*(255-255))/255))/255 = 0
        self.assertTrue(np.all(result_array == 0))

    def test_specific_color_blend(self):
        """Test softlight blending with specific color values."""
        # Create images with specific colors for predictable results
        red = Image.new('RGB', (1, 1), color=(100, 0, 0))
        green = Image.new('RGB', (1, 1), color=(0, 100, 0))
        
        result = imaging(red, green)
        result_array = np.array(result)
        
        # Calculate expected value manually
        # For red channel: in1=100, in2=0
        # ((255-100)*(100*0))/65536 + (100*(255-((255-100)*(255-0))/255))/255
        # = 0 + (100*(255-(155*255)/255))/255
        # = 0 + (100*(255-155))/255
        # = 0 + (100*100)/255
        # = 39 (approximately)
        
        # The other channels should be 0
        self.assertAlmostEqual(result_array[0, 0, 0], 39, delta=1)
        self.assertEqual(result_array[0, 0, 1], 0)
        self.assertEqual(result_array[0, 0, 2], 0)

if __name__ == '__main__':
    unittest.main()