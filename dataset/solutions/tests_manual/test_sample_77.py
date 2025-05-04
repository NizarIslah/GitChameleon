import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_77 import custom_product


def test_custom_product():
    arr = np.array([1, 2, 3, 4])
    result = custom_product(arr)
    expected = np.product(arr)
    assert result == expected


test_custom_product()
