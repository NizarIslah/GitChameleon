import unittest
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_88

CTYPE = ctypes.c_double
VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
c_array = create_c_array(VALUES, CTYPE)
assert all(isinstance(i, float) for i in c_array)
assert all(c_array[i] == VALUES[i] for i in range(len(VALUES)))
