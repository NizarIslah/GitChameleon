import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np
from PIL import Image, ImageChops

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_321 import imaging


def generate_random_image(width, height):
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(random_data)


width, height = 256, 256
img1 = generate_random_image(width, height)
img2 = generate_random_image(width, height)

gt = ImageChops.soft_light(img1, img2)
sol = imaging(img1, img2)
assert np.allclose(np.array(sol), np.array(sol))
