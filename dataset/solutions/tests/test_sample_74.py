import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_74 import custom_sometrue

class TestCustomSometrue(unittest.TestCase):
    
    def test_array_with_true_values(self):
        """Test that the function returns True when array contains True values."""
        # Test with array containing some True values
        arr = np.array([False, True, False])
        self.assertTrue(custom_sometrue(arr))
        
        # Test with array containing all True values
        arr = np.array([True, True, True])
        self.assertTrue(custom_sometrue(arr))
        
        # Test with numeric array containing non-zero values
        arr = np.array([0, 1, 2])
        self.assertTrue(custom_sometrue(arr))
    
    def test_array_with_no_true_values(self):
        """Test that the function returns False when array contains no True values."""
        # Test with array containing all False values
        arr = np.array([False, False, False])
        self.assertFalse(custom_sometrue(arr))
        
        # Test with numeric array containing all zeros
        arr = np.array([0, 0, 0])
        self.assertFalse(custom_sometrue(arr))
    
    def test_empty_array(self):
        """Test that the function returns False for an empty array."""
        arr = np.array([])
        self.assertFalse(custom_sometrue(arr))
    
    def test_multidimensional_array(self):
        """Test that the function works with multidimensional arrays."""
        # 2D array with some True values
        arr = np.array([[False, True], [False, False]])
        self.assertTrue(custom_sometrue(arr))
        
        # 2D array with no True values
        arr = np.array([[False, False], [False, False]])
        self.assertFalse(custom_sometrue(arr))
        
        # 3D array with some True values
        arr = np.array([[[False, False], [True, False]], [[False, False], [False, False]]])
        self.assertTrue(custom_sometrue(arr))
    
    def test_different_dtypes(self):
        """Test that the function works with arrays of different data types."""
        # Float array
        arr = np.array([0.0, 0.1, 0.0])
        self.assertTrue(custom_sometrue(arr))
        
        # String array (non-empty strings are truthy)
        arr = np.array(['', 'test', ''])
        self.assertTrue(custom_sometrue(arr))
        
        # Boolean masked array
        arr = np.ma.array([1, 2, 3], mask=[True, False, True])
        self.assertTrue(custom_sometrue(arr))

if __name__ == '__main__':
    unittest.main()