import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_74 import custom_sometrue


def test_custom_sometrue():
    arr = np.array([0, 0, 1, 0])
    result = custom_sometrue(arr)
    expected = np.any(arr)
    assert result == expected


test_custom_sometrue()
