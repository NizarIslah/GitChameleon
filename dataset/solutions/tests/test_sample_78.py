import pytest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_78 import custom_cumproduct


def test_custom_cumproduct_1d_array():
    """Test custom_cumproduct with a 1D array."""
    arr = np.array([1, 2, 3, 4])
    expected = np.array([1, 2, 6, 24])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)


def test_custom_cumproduct_empty():
    """Test custom_cumproduct with an empty array."""
    arr = np.array([])
    expected = np.array([])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)


def test_custom_cumproduct_2d_array():
    """Test custom_cumproduct with a 2D array (flattens then cumprod)."""
    arr = np.array([[1, 2], [3, 4]])
    # custom_cumproduct flattens the input before computing the cumulative product
    expected = arr.flatten().cumprod()
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)


def test_custom_cumproduct_negative_and_zero():
    """Test custom_cumproduct with negative values and zeros."""
    arr = np.array([2, -1, 0, 3])
    expected = np.array([2, -2, 0, 0])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)
