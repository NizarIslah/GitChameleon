import pytest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_78 import custom_cumproduct

def test_custom_cumproduct_basic():
    """Test custom_cumproduct with basic array."""
    arr = np.array([1, 2, 3, 4])
    expected = np.array([1, 2, 6, 24])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)

def test_custom_cumproduct_with_zeros():
    """Test custom_cumproduct with array containing zeros."""
    arr = np.array([2, 0, 3, 4])
    expected = np.array([2, 0, 0, 0])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)

def test_custom_cumproduct_with_negative():
    """Test custom_cumproduct with array containing negative numbers."""
    arr = np.array([1, -2, 3, -4])
    expected = np.array([1, -2, -6, 24])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)

def test_custom_cumproduct_empty_array():
    """Test custom_cumproduct with empty array."""
    arr = np.array([])
    expected = np.array([])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)

def test_custom_cumproduct_single_element():
    """Test custom_cumproduct with single element array."""
    arr = np.array([5])
    expected = np.array([5])
    result = custom_cumproduct(arr)
    np.testing.assert_array_equal(result, expected)

def test_custom_cumproduct_2d_array():
    """Test custom_cumproduct with 2D array."""
    arr = np.array([[1, 2], [3, 4]])
    # For 2D arrays, cumproduct operates along the specified axis
    # Testing axis=0 (default)
    expected_axis0 = np.array([[1, 2], [3, 8]])
    result_axis0 = custom_cumproduct(arr)
    np.testing.assert_array_equal(result_axis0, expected_axis0)