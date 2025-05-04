# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_49
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.utils import Bunch

data, _ = load_digits(return_X_y=True)
n_components = 7
expected_shape = (1797, n_components)
assert apply_fast_ica(data, n_components).shape == expected_shape
