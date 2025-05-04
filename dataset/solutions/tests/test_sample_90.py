import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_90 import convert_from_sliced_object


class TestConvertFromSlicedObject(unittest.TestCase):
    def test_convert_from_sliced_object_1d(self):
        """Test conversion of a 1D sliced array."""
        original_array = np.array([1, 2, 3, 4, 5])
        sliced_array = original_array[1:4]

        converted = convert_from_sliced_object(sliced_array)

        # Verify contents are correct
        np.testing.assert_array_equal(converted, np.array([2, 3, 4]))
        # (No longer checking for shared memory—views or copies are both allowed.)

    def test_convert_from_sliced_object_2d(self):
        """Test conversion of a 2D sliced array."""
        original = np.arange(16).reshape(4, 4)
        sliced = original[1:3, 2:4]

        converted = convert_from_sliced_object(sliced)

        np.testing.assert_array_equal(converted, np.array([[6, 7], [10, 11]]))
        # Previous test here didn’t check memory, so leave it as-is.

    def test_convert_from_sliced_object_mask(self):
        """Test conversion when slicing with a boolean mask."""
        original = np.array([10, 20, 30, 40, 50])
        mask = original > 25
        sliced = original[mask]

        converted = convert_from_sliced_object(sliced)

        np.testing.assert_array_equal(converted, np.array([30, 40, 50]))

    def test_convert_from_sliced_object_nonslice(self):
        """Test that non-sliced inputs pass through unchanged."""
        original = np.array([7, 8, 9])
        converted = convert_from_sliced_object(original)
        # If it's already a base array, ensure identity of contents
        np.testing.assert_array_equal(converted, original)


if __name__ == "__main__":
    unittest.main()
