import os
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_127 import apply_gaussian_filter1d


x = np.random.rand(100)
radius = 10
sigma = np.pi
output = apply_gaussian_filter1d(x, radius=radius, sigma=sigma)
assertion_value = np.allclose(
    output, gaussian_filter1d(x, truncate=radius / sigma, sigma=sigma)
)
assert assertion_value
