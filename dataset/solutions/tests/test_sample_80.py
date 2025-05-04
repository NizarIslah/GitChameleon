import sys
import os
import numpy as np
import pytest

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_80 import custom_alltrue


def test_custom_alltrue_all_true():
    """Test when all elements are True."""
    arr = np.array([True, True, True])
    result = custom_alltrue(arr)
    assert result == True


def test_custom_alltrue_some_false():
    """Test when some elements are False."""
    arr = np.array([True, False, True])
    result = custom_alltrue(arr)
    assert result == False


def test_custom_alltrue_all_false():
    """Test when all elements are False."""
    arr = np.array([False, False, False])
    result = custom_alltrue(arr)
    assert result == False


def test_custom_alltrue_empty_array():
    """Test with an empty array."""
    arr = np.array([])
    result = custom_alltrue(arr)
    # numpy.alltrue returns True for empty arrays
    assert result == True


def test_custom_alltrue_numeric_values():
    """Test with numeric values (non-zero is True, zero is False)."""
    arr = np.array([1, 2, 3])
    result = custom_alltrue(arr)
    assert result == True

    arr = np.array([0, 1, 2])
    result = custom_alltrue(arr)
    assert result == False


def test_custom_alltrue_mixed_types():
    """Test with mixed boolean and numeric values."""
    arr = np.array([True, 1, 5])
    result = custom_alltrue(arr)
    assert result == True

    arr = np.array([True, 0, 5])
    result = custom_alltrue(arr)
    assert result == False
