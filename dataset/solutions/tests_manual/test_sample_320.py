import unittest
import numpy as np
from PIL import Image, ImageChops
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_320 import imaging

import numpy as np
from PIL import Image, ImageChops


def generate_random_image(width, height):
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(random_data)


width, height = 256, 256
img1 = generate_random_image(width, height)
img2 = generate_random_image(width, height)

gt = ImageChops.overlay(img1, img2)
sol = imaging(img1, img2)
assert np.allclose(np.array(gt), np.array(sol))
