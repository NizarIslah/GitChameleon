import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_66 import apply_convolution_full

class TestApplyConvolutionFull(unittest.TestCase):
    
    def test_basic_convolution(self):
        """Test basic convolution with simple arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([0, 1, 0.5])
        expected = np.array([0, 1, 2.5, 4, 1.5])
        result = apply_convolution_full(arr1, arr2)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_empty_arrays(self):
        """Test convolution with empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([1, 2, 3])
        result = apply_convolution_full(arr1, arr2)
        self.assertEqual(result.size, 0)
    
    def test_single_element_arrays(self):
        """Test convolution with single element arrays."""
        arr1 = np.array([5])
        arr2 = np.array([2])
        expected = np.array([10])
        result = apply_convolution_full(arr1, arr2)
        np.testing.assert_array_equal(result, expected)
    
    def test_different_size_arrays(self):
        """Test convolution with arrays of different sizes."""
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([0.5, 0.5])
        expected = np.array([0.5, 1.5, 2.5, 3.5, 2])
        result = apply_convolution_full(arr1, arr2)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_compare_with_direct_numpy(self):
        """Test that our function matches NumPy's direct implementation."""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([5, 4, 3, 2, 1])
        expected = np.convolve(arr1, arr2, mode="full")
        result = apply_convolution_full(arr1, arr2)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()