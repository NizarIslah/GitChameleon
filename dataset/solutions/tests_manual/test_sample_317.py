import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_317 import imaging


def generate_random_image(width, height):
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(random_data)


def create(imIn1, imIn2, mode=None):
    if imIn1.shape != imIn2.shape:
        return None
    return np.empty_like(imIn1, dtype=np.uint8)


def imaging_overlay(imIn1, imIn2):
    imOut = create(imIn1, imIn2)
    if imOut is None:
        return None

    ysize, xsize, _ = imOut.shape
    for y in range(ysize):
        for x in range(xsize):
            for c in range(3):  # Loop over RGB channels
                in1, in2 = int(imIn1[y, x, c]), int(imIn2[y, x, c])
                if in1 < 128:
                    imOut[y, x, c] = np.clip((in1 * in2) // 127, 0, 255)
                else:
                    imOut[y, x, c] = np.clip(
                        255 - (((255 - in1) * (255 - in2)) // 127), 0, 255
                    )

    return imOut


width, height = 256, 256
img1 = generate_random_image(width, height)
img2 = generate_random_image(width, height)
gt = imaging_overlay(np.array(img1), np.array(img2))
sol = imaging(img1, img2)
assert np.allclose(np.array(gt), np.array(sol))
