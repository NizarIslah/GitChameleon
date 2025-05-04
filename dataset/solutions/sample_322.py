# library: pillow
# version: 7.1.0
# extra_dependencies: ['numpy==1.16']
import numpy as np
from PIL import Image, ImageChops


def imaging(img1: Image, img2: Image) -> Image:
    return ImageChops.hard_light(img1, img2)
