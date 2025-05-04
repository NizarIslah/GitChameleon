# library: pillow
# version: 7.0.0
# extra_dependencies: ['numpy==1.16']
import numpy as np
from PIL import Image, ImageChops


def imaging(img1: Image, img2: Image) -> Image:
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

    return imaging_softlight(np.array(img1), np.array(img2))
