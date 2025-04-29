import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_70 import find_common_type

class TestFindCommonType(unittest.TestCase):
    
    def test_same_dtype(self):
        """Test with arrays of the same data type."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.int32)
        result = find_common_type(arr1, arr2)
        self.assertEqual(result, np.int32)
    
    def test_different_dtypes(self):
        """Test with arrays of different data types."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = find_common_type(arr1, arr2)
        self.assertEqual(result, np.float64)
    
    def test_complex_dtype(self):
        """Test with complex data type."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1+2j, 3+4j], dtype=np.complex128)
        result = find_common_type(arr1, arr2)
        self.assertEqual(result, np.complex128)
    
    def test_bool_and_int(self):
        """Test with boolean and integer arrays."""
        arr1 = np.array([True, False, True], dtype=np.bool_)
        arr2 = np.array([1, 2, 3], dtype=np.int32)
        result = find_common_type(arr1, arr2)
        self.assertEqual(result, np.int32)
    
    def test_uint_and_int(self):
        """Test with unsigned and signed integer arrays."""
        arr1 = np.array([1, 2, 3], dtype=np.uint8)
        arr2 = np.array([4, 5, 6], dtype=np.int8)
        result = find_common_type(arr1, arr2)
        # The common type should be int16 to accommodate both uint8 and int8
        self.assertEqual(result, np.int16)

if __name__ == '__main__':
    unittest.main()