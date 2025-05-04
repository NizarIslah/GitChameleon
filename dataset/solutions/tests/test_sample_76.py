import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_76 import custom_round


class TestCustomRound(unittest.TestCase):
    def test_custom_round_integers(self):
        """Test that integers remain unchanged."""
        arr = np.array([1, 2, 3, 4, 5])
        result = custom_round(arr)
        np.testing.assert_array_equal(result, arr)

    def test_custom_round_decimals(self):
        """Test rounding of decimal values."""
        arr = np.array([1.4, 2.5, 3.6, 4.5, 5.5])
        expected = np.array([1.0, 2.0, 4.0, 4.0, 6.0])
        result = custom_round(arr)
        np.testing.assert_array_equal(result, expected)

    def test_custom_round_negative(self):
        """Test rounding of negative values."""
        arr = np.array([-1.4, -2.5, -3.6, -4.5, -5.5])
        expected = np.array([-1.0, -2.0, -4.0, -4.0, -6.0])
        result = custom_round(arr)
        np.testing.assert_array_equal(result, expected)

    def test_custom_round_mixed(self):
        """Test rounding of mixed positive and negative values."""
        arr = np.array([-1.7, 0, 1.2, 2.5, 3.7])
        expected = np.array([-2.0, 0, 1.0, 2.0, 4.0])
        result = custom_round(arr)
        np.testing.assert_array_equal(result, expected)

    def test_custom_round_empty(self):
        """Test rounding of an empty array."""
        arr = np.array([])
        result = custom_round(arr)
        np.testing.assert_array_equal(result, arr)

    def test_custom_round_multidimensional(self):
        """Test rounding of a multidimensional array."""
        arr = np.array([[1.4, 2.5], [3.6, 4.5]])
        expected = np.array([[1.0, 2.0], [4.0, 4.0]])
        result = custom_round(arr)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
