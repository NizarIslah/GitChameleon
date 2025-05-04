import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_318 import imaging


def generate_random_image(width, height):
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(random_data)


def create(imIn1, imIn2, mode=None):
    if imIn1.shape != imIn2.shape:
        return None
    return np.empty_like(imIn1, dtype=np.uint8)


def imaging_softlight(imIn1, imIn2):
    if imIn1.shape != imIn2.shape:
        return None

    imOut = create(imIn1, imIn2)
    ysize, xsize, _ = imOut.shape
    for y in range(ysize):
        for x in range(xsize):
            for c in range(3):  # Loop over RGB channels
                in1, in2 = int(imIn1[y, x, c]), int(imIn2[y, x, c])
                imOut[y, x, c] = int(
                    (((255 - in1) * (in1 * in2)) // 65536)
                    + (in1 * (255 - ((255 - in1) * (255 - in2) // 255))) // 255
                )
    return imOut


width, height = 256, 256
img1 = generate_random_image(width, height)
img2 = generate_random_image(width, height)


gt = imaging_softlight(np.array(img1), np.array(img2))
sol = imaging(img1, img2)
assert np.allclose(np.array(gt), np.array(sol))
