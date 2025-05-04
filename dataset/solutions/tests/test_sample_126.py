import os

# Add the directory containing sample_126.py to the Python path
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "solutions"))

# Import the function to test
from sample_126 import compute_lanczos_window


class TestLanczosWindow(unittest.TestCase):
    def test_window_size(self):
        """Test that the window has the correct size."""
        sizes = [5, 10, 15, 20]
        for size in sizes:
            window = compute_lanczos_window(size)
            self.assertEqual(len(window), size, f"Window size should be {size}")

    def test_normalization(self):
        """Test that the window is properly normalized (max value is 1.0)."""
        sizes = [5, 10, 15, 20]
        for size in sizes:
            window = compute_lanczos_window(size)
            self.assertAlmostEqual(
                np.max(window),
                1.0,
                msg="Window should be normalized with max value of 1.0",
            )

    def test_symmetry_odd_size(self):
        """Test that windows with odd sizes are symmetric."""
        sizes = [5, 11, 21]
        for size in sizes:
            window = compute_lanczos_window(size)
            mid_point = size // 2
            # Check if the first half mirrors the second half
            for i in range(mid_point):
                self.assertAlmostEqual(
                    window[i],
                    window[size - 1 - i],
                    places=10,
                    msg=f"Window should be symmetric at indices {i} and {size-1-i}",
                )

    def test_specific_values(self):
        """Test specific known values for the Lanczos window."""
        # Test with a small window size where we can manually verify values
        window_size = 5
        window = compute_lanczos_window(window_size)

        # For a window of size 5, the values should be approximately:
        # The middle value should be 1.0 (after normalization)
        # The first and last values should be the same due to symmetry
        self.assertAlmostEqual(
            window[2],
            1.0,
            places=10,
            msg="Center value should be 1.0 after normalization",
        )
        self.assertAlmostEqual(
            window[0], window[4], places=10, msg="First and last values should be equal"
        )
        self.assertAlmostEqual(
            window[1],
            window[3],
            places=10,
            msg="Second and fourth values should be equal",
        )

        # The sinc function at x=0 is 1, so the middle value before normalization should be 1
        # After normalization, it remains 1
        # For x = ±1, sinc(x) = sin(π*x)/(π*x) = 0
        # But we're not exactly at ±1 for the first and last elements, so they're small but non-zero
        self.assertGreater(window[0], 0, msg="First value should be positive but small")
        self.assertLess(window[0], 0.3, msg="First value should be less than 0.3")

    def test_edge_cases(self):
        """Test edge cases for the Lanczos window."""
        # Test with minimum valid window size (2)
        window = compute_lanczos_window(2)
        self.assertEqual(len(window), 2, "Window size should be 2")
        self.assertAlmostEqual(
            np.max(window), 1.0, msg="Window should be normalized with max value of 1.0"
        )

        # Test with a large window size
        large_size = 1001
        window = compute_lanczos_window(large_size)
        self.assertEqual(len(window), large_size, f"Window size should be {large_size}")
        self.assertAlmostEqual(
            np.max(window), 1.0, msg="Window should be normalized with max value of 1.0"
        )


if __name__ == "__main__":
    unittest.main()
