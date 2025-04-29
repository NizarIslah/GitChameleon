import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_69 import find_common_type

class TestFindCommonType(unittest.TestCase):
    
    def test_find_common_type_same_types(self):
        """Test when both arrays have the same data type."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.int32)
        
        # This will fail due to the bug in the implementation (array1 vs arr1)
        # But this is how it should work when fixed
        try:
            result = find_common_type(arr1, arr2)
            self.assertEqual(result, np.dtype('int32'))
        except NameError:
            self.fail("Function uses incorrect variable names (likely 'array1' instead of 'arr1')")
    
    def test_find_common_type_different_types(self):
        """Test when arrays have different data types."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        
        try:
            result = find_common_type(arr1, arr2)
            self.assertEqual(result, np.dtype('float64'))
        except NameError:
            self.fail("Function uses incorrect variable names (likely 'array1' instead of 'arr1')")
    
    def test_find_common_type_complex(self):
        """Test with complex numbers."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1+2j, 3+4j], dtype=np.complex128)
        
        try:
            result = find_common_type(arr1, arr2)
            self.assertEqual(result, np.dtype('complex128'))
        except NameError:
            self.fail("Function uses incorrect variable names (likely 'array1' instead of 'arr1')")

if __name__ == '__main__':
    unittest.main()