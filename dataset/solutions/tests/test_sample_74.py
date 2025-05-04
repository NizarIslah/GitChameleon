import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_74 import custom_sometrue


class TestCustomSometrue(unittest.TestCase):
    def test_integer_array(self):
        """Test with an integer array."""
        arr = np.array([0, 0, 1], dtype=int)
        self.assertTrue(custom_sometrue(arr))
        arr = np.array([0, 0, 0], dtype=int)
        self.assertFalse(custom_sometrue(arr))

    def test_boolean_array(self):
        """Test with a boolean array."""
        arr = np.array([False, False, True], dtype=bool)
        self.assertTrue(custom_sometrue(arr))
        arr = np.array([False, False, False], dtype=bool)
        self.assertFalse(custom_sometrue(arr))

    def test_different_dtypes(self):
        """Test with a float array (works), and a string array (raises)."""
        # Float array should work
        arr = np.array([0.0, 0.1, 0.0], dtype=float)
        self.assertTrue(custom_sometrue(arr))

        # String array will cause a ufunc error under np.any → we expect a TypeError
        arr_str = np.array(["", "test", ""], dtype="<U4")
        with self.assertRaises(TypeError):
            custom_sometrue(arr_str)

    def test_empty_array(self):
        """Test with an empty array."""
        arr = np.array([], dtype=int)
        # np.any on empty numeric returns False
        self.assertFalse(custom_sometrue(arr))

    def test_object_array(self):
        """Test with an object-dtype array of strings."""
        arr = np.array(["", "test", ""], dtype=object)
        # np.any on object dtype falls back, returns 'test', which is truthy
        self.assertTrue(custom_sometrue(arr))


if __name__ == "__main__":
    unittest.main()
