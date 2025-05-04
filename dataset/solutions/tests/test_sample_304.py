import os

# Add the dataset/samples directory to the Python path
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "samples")
    ),
)

# Import the function to test
from sample_304 import compute_localmin


class TestComputeLocalmin(unittest.TestCase):
    """Test cases for the compute_localmin function."""

    def test_1d_array(self):
        """Test compute_localmin with a 1D array."""
        # Create a test array with known local minima
        x = np.array([1, 0, 1, 2, -1, 0, -2, 1])

        # Expected result: local minima at indices 1, 4, and 6
        expected = np.array([False, True, False, False, True, False, True, False])

        # Test with axis=0
        result = compute_localmin(x, axis=0)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_axis0(self):
        """Test compute_localmin with a 2D array along axis 0."""
        # Create a 2D test array
        x = np.array([[1, 0, 1], [2, -1, 0], [2, 1, 3]])

        # Expected result: local minima along axis 0
        expected = np.array(
            [[False, False, False], [False, True, True], [False, False, False]]
        )

        # Test with axis=0
        result = compute_localmin(x, axis=0)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_axis1(self):
        """Test compute_localmin with a 2D array along axis 1."""
        # Create a 2D test array
        x = np.array([[1, 0, 1], [2, -1, 0], [2, 1, 3]])

        # Expected result: local minima along axis 1
        expected = np.array(
            [[False, True, False], [False, True, False], [False, True, False]]
        )

        # Test with axis=1
        result = compute_localmin(x, axis=1)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test compute_localmin with an empty array."""
        # Create an empty array
        x = np.array([])

        # Expected result: empty array
        expected = np.array([], dtype=bool)

        # Test with axis=0
        result = compute_localmin(x, axis=0)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def test_constant_array(self):
        """Test compute_localmin with a constant array."""
        # Create a constant array
        x = np.ones(5)

        # Expected result: no local minima
        expected = np.zeros(5, dtype=bool)

        # Test with axis=0
        result = compute_localmin(x, axis=0)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def test_edge_cases(self):
        """Test compute_localmin with edge cases."""
        # Test with a single element array
        x_single = np.array([5])
        expected_single = np.array([False])
        result_single = compute_localmin(x_single, axis=0)
        np.testing.assert_array_equal(result_single, expected_single)

        # Test with an array where the first element is the minimum
        x_first = np.array([0, 1, 2, 3])
        expected_first = np.array(
            [False, False, False, False]
        )  # First element is never a local minimum
        result_first = compute_localmin(x_first, axis=0)
        np.testing.assert_array_equal(result_first, expected_first)

        # Test with an array where the last element is the minimum
        x_last = np.array([3, 2, 1, 0])
        expected_last = np.array(
            [False, False, False, True]
        )  # Last element can be a local minimum
        result_last = compute_localmin(x_last, axis=0)
        np.testing.assert_array_equal(result_last, expected_last)


if __name__ == "__main__":
    unittest.main()
