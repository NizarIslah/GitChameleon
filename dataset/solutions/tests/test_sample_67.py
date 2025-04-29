import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_67 import apply_convolution_valid

class TestApplyConvolutionValid(unittest.TestCase):
    
    def test_basic_convolution(self):
        """Test basic convolution with simple arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([0, 1, 0.5])
        expected = np.array([1, 2.5, 3.5, 1.5])
        result = apply_convolution_valid(arr1, arr2)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_empty_result(self):
        """Test convolution that results in empty array due to 'valid' mode."""
        arr1 = np.array([1, 2])
        arr2 = np.array([1, 2, 3])
        # In 'valid' mode, output size is max(M, N) - min(M, N) + 1
        # Here it would be 3 - 2 + 1 = 2, but since arr2 is longer, result is empty
        expected = np.array([])
        result = apply_convolution_valid(arr1, arr2)
        self.assertEqual(result.size, 0)
    
    def test_same_size_arrays(self):
        """Test convolution with arrays of the same size."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        expected = np.array([4, 13, 28, 27, 18])
        result = apply_convolution_valid(arr1, arr2)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_different_dtypes(self):
        """Test convolution with arrays of different data types."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([0.5, 1.5], dtype=np.float64)
        expected = np.array([0.5, 2.5, 5.5])
        result = apply_convolution_valid(arr1, arr2)
        np.testing.assert_array_almost_equal(result, expected)
        # Result should be float64 as per NumPy's type promotion rules
        self.assertEqual(result.dtype, np.float64)

if __name__ == '__main__':
    unittest.main()