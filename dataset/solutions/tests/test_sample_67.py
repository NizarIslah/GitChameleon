import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_67 import apply_convolution_valid


class TestApplyConvolutionValid(unittest.TestCase):
    def test_basic_convolution(self):
        """Test basic 'valid' convolution with simple arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([0, 1, 0.5])
        result = apply_convolution_valid(arr1, arr2)
        expected = np.convolve(arr1, arr2, mode="valid")
        np.testing.assert_array_almost_equal(result, expected)

    def test_different_dtypes(self):
        """Test 'valid' convolution with arrays of different dtypes."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([0.5, 1.5], dtype=np.float64)
        result = apply_convolution_valid(arr1, arr2)
        expected = np.convolve(arr1, arr2, mode="valid")
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_arrays(self):
        """Test that passing an empty array raises ValueError."""
        arr1 = np.array([])
        arr2 = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            apply_convolution_valid(arr1, arr2)
        # Also check the other order
        with self.assertRaises(ValueError):
            apply_convolution_valid(arr2, arr1)

    def test_same_size_arrays(self):
        """Test 'valid' convolution when both arrays have equal length."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = apply_convolution_valid(arr1, arr2)
        expected = np.convolve(arr1, arr2, mode="valid")
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
