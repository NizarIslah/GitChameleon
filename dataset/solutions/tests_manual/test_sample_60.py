import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_60 import get_slice

ser = pd.Series([1, 2, 3, 4, 5], index=[2, 3, 5, 7, 11])
start, end = 2, 4
sliced_ser = ser[2:4]
assert sliced_ser.equals(
    get_slice(ser, start, end)
), "Slicing does not match expected output"
