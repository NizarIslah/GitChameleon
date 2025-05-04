# Test for sample_89.py
import unittest
import ctypes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_89 import c_str

python_string = "lightgbm"
c_string = c_str(python_string)
assert isinstance(c_string, ctypes.c_char_p)
assert c_string.value.decode("utf-8") == python_string
