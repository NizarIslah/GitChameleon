# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_302 import compute_shear


class TestComputeShear(unittest.TestCase):
    def test_compute_shear_2d_array(self):
        """Test compute_shear with a 2D array."""
        # Create a simple 2D array for testing
        test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Test with factor=1, axis=0 (shear along rows)
        result = compute_shear(test_array, factor=1, axis=0)

        # Expected result after shearing with factor=1, axis=0
        # Each row i is shifted right by i positions (with wrap)
        expected = np.array([[1, 2, 3], [6, 4, 5], [8, 9, 7]])

        np.testing.assert_array_equal(result, expected)

    def test_compute_shear_2d_array_axis_1(self):
        """Test compute_shear with a 2D array along axis 1."""
        # Create a simple 2D array for testing
        test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Test with factor=1, axis=1 (shear along columns)
        result = compute_shear(test_array, factor=1, axis=1)

        # Expected result after shearing with factor=1, axis=1 (with wrap)
        expected = np.array([[1, 8, 6], [4, 2, 9], [7, 5, 3]])

        np.testing.assert_array_equal(result, expected)

    def test_compute_shear_negative_factor(self):
        """Test compute_shear with a negative factor."""
        # Create a simple 2D array for testing
        test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Test with factor=-1, axis=0 (shear along rows in opposite direction)
        result = compute_shear(test_array, factor=-1, axis=0)

        # Expected result after shearing with factor=-1, axis=0 (with wrap)
        expected = np.array([[1, 2, 3], [5, 6, 4], [9, 7, 8]])

        np.testing.assert_array_equal(result, expected)

    def test_compute_shear_3d_array(self):
        """Test compute_shear with a 3D array."""
        # Create a simple 3D array for testing
        test_array = np.ones((2, 3, 2))

        # Test with factor=1, axis=1
        result = compute_shear(test_array, factor=1, axis=1)

        # The shape should remain the same
        self.assertEqual(result.shape, test_array.shape)

        # Verify the function doesn't raise exceptions with 3D arrays
        # and returns a result with the same shape


if __name__ == "__main__":
    unittest.main()
