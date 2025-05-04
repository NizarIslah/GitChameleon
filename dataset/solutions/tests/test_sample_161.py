# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_161 import compute_hilbert_transform


class TestHilbertTransform(unittest.TestCase):
    def test_basic_functionality(self):
        # Test with simple arrays
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = compute_hilbert_transform(a, b)

        # Check shape - should be 2 rows (stacked a and b) and 3 columns
        self.assertEqual(result.shape, (2, 3))

        # Check dtype - should be complex128
        self.assertEqual(result.dtype, np.complex128)

        # Check that result is complex (Hilbert transform returns complex values)
        self.assertTrue(np.iscomplexobj(result))

        # Check that the real part of the result matches the input for the first row
        np.testing.assert_allclose(result[0].real, a)

        # Check that the real part of the result matches the input for the second row
        np.testing.assert_allclose(result[1].real, b)

    def test_different_dtypes(self):
        # Test with float32 dtype
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = compute_hilbert_transform(a, b, dtype=np.float32)

        # Check that dtype is respected
        self.assertEqual(
            result.dtype, np.complex64
        )  # Hilbert transform of float32 is complex64

        # Test with float64 dtype explicitly
        result_float64 = compute_hilbert_transform(a, b, dtype=np.float64)
        self.assertEqual(
            result_float64.dtype, np.complex128
        )  # Hilbert transform of float64 is complex128

    def test_safe_casting(self):
        # Test with arrays that require safe casting
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4.5, 5.5, 6.5], dtype=np.float64)

        # This should work with safe casting
        result = compute_hilbert_transform(a, b)
        self.assertEqual(result.shape, (2, 3))

        # Check that the values were properly converted
        np.testing.assert_allclose(result[0].real, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1].real, [4.5, 5.5, 6.5])

    def test_type_error(self):
        # Test with arrays that cannot be safely cast
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4.5, 5.5, 6.5], dtype=np.float64)

        # This should raise TypeError because we're trying to cast float64 to int32
        with self.assertRaises(TypeError):
            compute_hilbert_transform(a, b, dtype=np.int32)

    def test_different_shapes(self):
        # Test with arrays of different lengths
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6])

        # This should raise ValueError because arrays have different shapes
        with self.assertRaises(ValueError):
            compute_hilbert_transform(a, b)

    def test_multidimensional_arrays(self):
        # Test with 2D arrays
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = compute_hilbert_transform(a, b)

        # Check shape - should be 4 rows (stacked a and b flattened) and 2 columns
        self.assertEqual(result.shape, (4, 2))

        # Check that the real part matches the input
        np.testing.assert_allclose(result[0].real, a[0])
        np.testing.assert_allclose(result[1].real, a[1])
        np.testing.assert_allclose(result[2].real, b[0])
        np.testing.assert_allclose(result[3].real, b[1])


if __name__ == "__main__":
    unittest.main()
