# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_47
from sklearn.datasets import make_sparse_coded_signal

n_samples = 100
n_features = 50
n_components = 20
n_nonzero_coefs = 10
expected_shape_y = (n_features, n_samples)
expected_shape_d = (n_features, n_components)
expected_shape_c = (n_components, n_samples)

y, d, c = get_signal(n_samples, n_features, n_components, n_nonzero_coefs)
assert y.shape == expected_shape_y
assert d.shape == expected_shape_d
assert c.shape == expected_shape_c
