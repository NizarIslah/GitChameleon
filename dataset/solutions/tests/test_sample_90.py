import unittest
import numpy as np
import lightgbm as lgb
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_90 import convert_from_sliced_object


class TestConvertFromSlicedObject(unittest.TestCase):
    """Test cases for the convert_from_sliced_object function."""

    def test_convert_from_sliced_object_1d(self):
        """Test conversion of a 1D sliced array."""
        # Create a 1D numpy array and slice it
        original_array = np.array([1, 2, 3, 4, 5])
        sliced_array = original_array[1:4]
        
        # Convert the sliced array
        converted_array = convert_from_sliced_object(sliced_array)
        
        # Check that the conversion worked correctly
        np.testing.assert_array_equal(converted_array, np.array([2, 3, 4]))
        
        # Verify it's a new array, not a view
        self.assertFalse(np.may_share_memory(converted_array, original_array))

    def test_convert_from_sliced_object_2d(self):
        """Test conversion of a 2D sliced array."""
        # Create a 2D numpy array and slice it
        original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sliced_array = original_array[0:2, 1:3]
        
        # Convert the sliced array
        converted_array = convert_from_sliced_object(sliced_array)
        
        # Check that the conversion worked correctly
        np.testing.assert_array_equal(converted_array, np.array([[2, 3], [5, 6]]))
        
        # Verify it's a new array, not a view
        self.assertFalse(np.may_share_memory(converted_array, original_array))

    def test_convert_from_sliced_object_non_sliced(self):
        """Test conversion of a non-sliced array."""
        # Create a regular numpy array (not sliced)
        original_array = np.array([1, 2, 3])
        
        # Convert the array
        converted_array = convert_from_sliced_object(original_array)
        
        # For a non-sliced array, the function should still return a valid array
        np.testing.assert_array_equal(converted_array, original_array)

    def test_convert_from_sliced_object_empty(self):
        """Test conversion of an empty array."""
        # Create an empty numpy array
        empty_array = np.array([])
        
        # Convert the empty array
        converted_array = convert_from_sliced_object(empty_array)
        
        # Check that the conversion worked correctly
        np.testing.assert_array_equal(converted_array, empty_array)


if __name__ == '__main__':
    unittest.main()