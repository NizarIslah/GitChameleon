import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_66 import apply_convolution_full


class TestApplyConvolutionFull(unittest.TestCase):
    def test_simple_case(self):
        """Test a basic convolution with known output."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([0, 1, 0.5])
        result = apply_convolution_full(arr1, arr2)
        expected = np.array([0.0, 1.0, 2.5, 4.0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_different_lengths(self):
        """Test convolution where the first array is longer."""
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([1, 1, 1])
        result = apply_convolution_full(arr1, arr2)
        expected = np.convolve(arr1, arr2, mode="full")
        np.testing.assert_array_equal(result, expected)

    def test_empty_arrays(self):
        """Test that convolution with an empty array raises ValueError."""
        arr1 = np.array([])
        arr2 = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            apply_convolution_full(arr1, arr2)

    def test_single_element_arrays(self):
        """Test convolution when both arrays have a single element."""
        arr1 = np.array([5])
        arr2 = np.array([2])
        result = apply_convolution_full(arr1, arr2)
        expected = np.array([10])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
