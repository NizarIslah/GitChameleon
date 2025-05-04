import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_72 import custom_product


def test_custom_product():
    arr = np.array([1, 2, 3, 4])
    result = custom_product(arr)
    expected = np.prod(arr)
    assert result == expected


test_custom_product()
