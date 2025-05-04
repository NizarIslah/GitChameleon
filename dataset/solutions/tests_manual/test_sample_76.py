import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_76 import custom_round


def test_custom_round():
    arr = np.array([1.5, 2.3, 3.7])
    result = custom_round(arr)
    expected = np.round_(arr)
    assert np.array_equal(result, expected)


test_custom_round()
