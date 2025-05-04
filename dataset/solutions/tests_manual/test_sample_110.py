# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.spatial import distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_110 import compute_wminkowski


u = np.asarray([11, 12, 13, 14, 15])
v = np.asarray([1, 2, 3, 4, 5])
w = np.asarray([0.1, 0.3, 0.15, 0.25, 0.2])
output = compute_wminkowski(u, v, p=3, w=w)
assertion_value = np.allclose(output, distance.minkowski(u, v, p=3, w=w))
assert assertion_value
