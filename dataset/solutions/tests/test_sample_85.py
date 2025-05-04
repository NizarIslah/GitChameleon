# Test file for sample_85.py
import unittest
import ctypes
import numpy as np
import lightgbm as lgb
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_85 import convert_cint32_array_to_numpy


class TestConvertCint32ArrayToNumpy(unittest.TestCase):
    def test_convert_cint32_array_to_numpy(self):
        # Create a C array of integers
        test_data = [1, 2, 3, 4, 5]
        length = len(test_data)

        # Create a ctypes array
        c_array = (ctypes.c_int32 * length)(*test_data)
        c_pointer = ctypes.cast(c_array, ctypes.POINTER(ctypes.c_int32))

        # Call the function to convert to numpy array
        result = convert_cint32_array_to_numpy(c_pointer, length)

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Verify the contents match the original data
        np.testing.assert_array_equal(result, np.array(test_data))

    def test_empty_array(self):
        # Test with an empty array
        length = 0
        c_array = (ctypes.c_int32 * length)()
        c_pointer = ctypes.cast(c_array, ctypes.POINTER(ctypes.c_int32))

        # Call the function to convert to numpy array
        result = convert_cint32_array_to_numpy(c_pointer, length)

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Verify the array is empty
        self.assertEqual(len(result), 0)

    def test_large_array(self):
        # Test with a larger array
        test_data = list(range(1000))
        length = len(test_data)

        # Create a ctypes array
        c_array = (ctypes.c_int32 * length)(*test_data)
        c_pointer = ctypes.cast(c_array, ctypes.POINTER(ctypes.c_int32))

        # Call the function to convert to numpy array
        result = convert_cint32_array_to_numpy(c_pointer, length)

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Verify the contents match the original data
        np.testing.assert_array_equal(result, np.array(test_data))


if __name__ == "__main__":
    unittest.main()
