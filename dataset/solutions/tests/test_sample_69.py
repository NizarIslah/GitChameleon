import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_69 import find_common_type


class TestFindCommonType(unittest.TestCase):
    def test_find_common_type_with_different_dtypes(self):
        # Create arrays with different data types
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4.5, 5.5, 6.5], dtype=np.float64)

        # The expected common type should be float64
        expected_dtype = np.float64

        # This will fail because of the bug in the original function
        # (it uses array1 instead of arr1)
        # But the test is correct
        result_dtype = find_common_type(arr1, arr2)

        self.assertEqual(result_dtype, expected_dtype)

    def test_find_common_type_with_complex(self):
        # Create arrays with complex and float types
        arr1 = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        arr2 = np.array([1.5, 2.5], dtype=np.float64)

        # The expected common type should be complex128
        expected_dtype = np.complex128

        result_dtype = find_common_type(arr1, arr2)

        self.assertEqual(result_dtype, expected_dtype)


if __name__ == "__main__":
    unittest.main()
