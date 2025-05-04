import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_79 import custom_anytrue


class TestCustomAnyTrue(unittest.TestCase):
    def test_all_true(self):
        """Test with an array where all elements are True."""
        arr = np.array([True, True, True])
        result = custom_anytrue(arr)
        self.assertTrue(result)

    def test_some_true(self):
        """Test with an array where some elements are True."""
        arr = np.array([False, True, False])
        result = custom_anytrue(arr)
        self.assertTrue(result)

    def test_none_true(self):
        """Test with an array where no elements are True."""
        arr = np.array([False, False, False])
        result = custom_anytrue(arr)
        self.assertFalse(result)

    def test_empty_array(self):
        """Test with an empty array."""
        arr = np.array([])
        result = custom_anytrue(arr)
        self.assertFalse(result)

    def test_numeric_array(self):
        """Test with a numeric array (non-zero values are considered True)."""
        arr = np.array([0, 1, 0])
        result = custom_anytrue(arr)
        self.assertTrue(result)

        arr = np.array([0, 0, 0])
        result = custom_anytrue(arr)
        self.assertFalse(result)

    def test_multidimensional_array(self):
        """Test with a multidimensional array."""
        arr = np.array([[False, False], [False, True]])
        result = custom_anytrue(arr)
        self.assertTrue(result)

        arr = np.array([[False, False], [False, False]])
        result = custom_anytrue(arr)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
