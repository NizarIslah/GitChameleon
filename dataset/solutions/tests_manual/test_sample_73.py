import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_73 import custom_cumproduct


def test_custom_cumproduct():
    arr = np.array([1, 2, 3, 4])
    result = custom_cumproduct(arr)
    expected = np.cumprod(arr)
    assert np.array_equal(result, expected)


test_custom_cumproduct()
