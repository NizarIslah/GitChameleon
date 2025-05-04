# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_45
from sklearn.cross_decomposition import CCA

X = np.random.rand(100, 10)
Y = np.random.rand(100, 5)
cca_model = CCA()
correct_shape = (X.shape[1], Y.shape[1])
assert get_coef_shape(cca_model, X, Y) == correct_shape
