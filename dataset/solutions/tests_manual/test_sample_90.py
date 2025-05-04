import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_90 import convert_from_sliced_object

data = np.random.rand(100, 10)
sliced_data = data[:, :5]
fixed_data = convert_from_sliced_object(sliced_data)
assert isinstance(fixed_data, np.ndarray)
assert fixed_data.shape == sliced_data.shape
assert np.array_equal(fixed_data, sliced_data)
