import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_68 import apply_correlate_full


class TestApplyCorrelateFull(unittest.TestCase):
    def test_empty_arrays(self):
        """Test that calling with two empty arrays raises ValueError."""
        arr1 = np.array([])
        arr2 = np.array([])
        with self.assertRaises(ValueError):
            apply_correlate_full(arr1, arr2)

    def test_single_element_arrays(self):
        """Test correlation when both arrays have a single element."""
        arr1 = np.array([5])
        arr2 = np.array([2])
        result = apply_correlate_full(arr1, arr2)
        expected = np.correlate(arr1, arr2, mode="full")
        np.testing.assert_array_equal(result, expected)

    def test_different_length_arrays(self):
        """Test correlation with arrays of different lengths."""
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([0, 1])
        result = apply_correlate_full(arr1, arr2)
        expected = np.correlate(arr1, arr2, mode="full")
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
