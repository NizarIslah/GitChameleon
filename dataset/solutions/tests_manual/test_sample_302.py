# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_302 import compute_shear


E = np.eye(3)
factor = -1
axis = -1

sol = compute_shear(E, factor, axis)
gt = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
assert np.array_equal(gt, sol)
