import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_61 import get_slice

ser = pd.Series([1, 2, 3, 4, 5], index=[2, 3, 5, 7, 11])
start, end = 2, 4
sliced_ser = ser.iloc[2:4]
assert sliced_ser.equals(
    get_slice(ser, start, end)
), "Slicing does not match expected label-based output"
