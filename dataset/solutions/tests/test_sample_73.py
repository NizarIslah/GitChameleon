import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_73 import custom_cumproduct


class TestCustomCumproduct(unittest.TestCase):
    def test_custom_cumproduct_1d_array(self):
        """Test with a 1D array."""
        arr = np.array([1, 2, 3, 4])
        expected = np.array([1, 2, 6, 24])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)

    def test_custom_cumproduct_2d_array(self):
        """Test with a 2D array."""
        arr = np.array([[1, 2], [3, 4]])
        # For multi-dimensional inputs, custom_cumproduct flattens and then computes
        expected = arr.cumprod().ravel()
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)

    def test_custom_cumproduct_empty(self):
        """Test with an empty array."""
        arr = np.array([])
        expected = np.array([])
        np.testing.assert_array_equal(custom_cumproduct(arr), expected)

    # ... any other existing tests remain unchanged ...


if __name__ == "__main__":
    unittest.main()
