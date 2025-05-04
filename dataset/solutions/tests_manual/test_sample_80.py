import sys
import os
import numpy as np
import pytest

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_80 import custom_alltrue


def test_custom_alltrue():
    arr = np.array([1, 1, 1, 1])
    result = custom_alltrue(arr)
    expected = np.alltrue(arr)
    assert result == expected


test_custom_alltrue()
