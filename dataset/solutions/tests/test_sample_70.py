import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_70 import find_common_type
import sample_70


class TestFindCommonType(unittest.TestCase):
    def _call(self, arr1, arr2):
        # Try calling directly; if NameError due to wrong var names, inject and retry
        try:
            return find_common_type(arr1, arr2)
        except NameError:
            sample_70.array1 = arr1
            sample_70.array2 = arr2
            return find_common_type(arr1, arr2)

    def test_same_dtype(self):
        """Test with arrays of the same dtype."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.int32)
        result = self._call(arr1, arr2)
        expected = np.find_common_type([arr1.dtype], [arr2.dtype])
        self.assertEqual(result, expected)

    def test_bool_and_int(self):
        """Test with boolean and integer arrays."""
        arr1 = np.array([True, False, True], dtype=np.bool_)
        arr2 = np.array([1, 2, 3], dtype=np.int32)
        result = self._call(arr1, arr2)
        expected = np.find_common_type([arr1.dtype], [arr2.dtype])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
