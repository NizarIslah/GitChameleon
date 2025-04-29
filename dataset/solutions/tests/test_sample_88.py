import unittest
import ctypes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_88

class TestCreateCArray(unittest.TestCase):
    def test_create_c_array_int(self):
        """Test creating a ctypes array with integer values."""
        # Test with integer values
        values = [1, 2, 3, 4, 5]
        ctype = ctypes.c_int
        
        # This will fail if the function has the bug using CTYPE and VALUES
        # instead of ctype and values
        result = sample_88.create_c_array(values, ctype)
        
        # Check that the result is a ctypes array
        self.assertTrue(isinstance(result, ctypes.Array))
        
        # Check that the array has the correct length
        self.assertEqual(len(result), len(values))
        
        # Check that each element has the correct value
        for i, val in enumerate(values):
            self.assertEqual(result[i], val)
    
    def test_create_c_array_float(self):
        """Test creating a ctypes array with float values."""
        # Test with float values
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        ctype = ctypes.c_float
        
        result = sample_88.create_c_array(values, ctype)
        
        # Check that the result is a ctypes array
        self.assertTrue(isinstance(result, ctypes.Array))
        
        # Check that the array has the correct length
        self.assertEqual(len(result), len(values))
        
        # Check that each element has approximately the correct value
        # (using almost equal because of potential floating point precision issues)
        for i, val in enumerate(values):
            self.assertAlmostEqual(result[i], val, places=5)
    
    def test_create_c_array_empty(self):
        """Test creating a ctypes array with an empty list."""
        values = []
        ctype = ctypes.c_int
        
        result = sample_88.create_c_array(values, ctype)
        
        # Check that the result is a ctypes array
        self.assertTrue(isinstance(result, ctypes.Array))
        
        # Check that the array has the correct length (0)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()