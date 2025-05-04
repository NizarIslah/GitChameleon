import os
import sys
import unittest

import numpy as np
from scipy.signal import hilbert

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_162 import compute_hilbert_transform


class TestHilbertTransform(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic functionality with simple arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        result = compute_hilbert_transform(a, b)

        # Expected: hilbert transform of stacked arrays
        expected = hilbert(np.vstack((a, b)))
        np.testing.assert_array_equal(result, expected)

        # Check output dtype
        self.assertEqual(result.dtype, np.complex128)

    def test_unsafe_casting(self):
        """Test that TypeError is raised for unsafe casting."""
        # Complex input can't be safely cast to float
        a = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)

        with self.assertRaises(TypeError):
            compute_hilbert_transform(a, b)

    def test_different_shapes(self):
        """Test with arrays of different shapes."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        b = np.array([[5.0, 6.0], [7.0, 8.0]])  # 2x2

        result = compute_hilbert_transform(a, b)

        # Expected: hilbert transform of stacked arrays
        expected = hilbert(np.vstack((a, b)))
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
