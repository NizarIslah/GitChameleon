# Test file for sample_85.py
import unittest
import ctypes
import numpy as np
import lightgbm as lgb
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_85 import convert_cint32_array_to_numpy

c_array_type = ctypes.POINTER(ctypes.c_int32)
c_array = (ctypes.c_int32 * 5)(1, 2, 3, 4, 5)
c_pointer = ctypes.cast(c_array, c_array_type)
length = 5
np_array = convert_cint32_array_to_numpy(c_pointer, length)
assert isinstance(np_array, np.ndarray)
assert np_array.shape == (5,)
assert np.array_equal(np_array, np.array([1, 2, 3, 4, 5], dtype=np.int32))
