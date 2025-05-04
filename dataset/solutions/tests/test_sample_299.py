# Test file for sample_299.py
import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_299 import compute_chirp


class TestComputeChirp(unittest.TestCase):
    """Test cases for the compute_chirp function."""

    def test_linear_chirp_shape(self):
        """Test that a linear chirp has the expected shape."""
        fmin = 100
        fmax = 1000
        duration = 2
        sr = 22050
        linear = True

        chirp = compute_chirp(fmin, fmax, duration, sr, linear)

        # Check that the shape is correct (duration * sr samples)
        expected_length = duration * sr
        self.assertEqual(
            len(chirp),
            expected_length,
            f"Expected chirp length {expected_length}, got {len(chirp)}",
        )

        # Check that the output is a numpy array
        self.assertIsInstance(chirp, np.ndarray, "Output should be a numpy array")

    def test_logarithmic_chirp_shape(self):
        """Test that a logarithmic chirp has the expected shape."""
        fmin = 100
        fmax = 1000
        duration = 1
        sr = 44100
        linear = False

        chirp = compute_chirp(fmin, fmax, duration, sr, linear)

        # Check that the shape is correct (duration * sr samples)
        expected_length = duration * sr
        self.assertEqual(
            len(chirp),
            expected_length,
            f"Expected chirp length {expected_length}, got {len(chirp)}",
        )

    def test_chirp_values_range(self):
        """Test that the chirp values are within the expected range."""
        fmin = 200
        fmax = 800
        duration = 0.5
        sr = 16000

        # Test both linear and logarithmic chirps
        for linear in [True, False]:
            chirp = compute_chirp(fmin, fmax, duration, sr, linear)

            # Chirp values should be between -1 and 1 (sine wave amplitude)
            self.assertTrue(
                np.all(chirp >= -1.0),
                f"Chirp values should be >= -1.0, found {np.min(chirp)}",
            )
            self.assertTrue(
                np.all(chirp <= 1.0),
                f"Chirp values should be <= 1.0, found {np.max(chirp)}",
            )

    def test_different_parameters(self):
        """Test the function with different parameter combinations."""
        test_cases = [
            # fmin, fmax, duration, sr, linear
            (20, 20000, 3, 8000, True),
            (500, 1500, 0.1, 48000, False),
            (50, 5000, 2, 22050, True),
            (100, 1000, 1, 16000, False),
        ]

        for fmin, fmax, duration, sr, linear in test_cases:
            chirp = compute_chirp(fmin, fmax, duration, sr, linear)
            expected_length = int(duration * sr)
            self.assertEqual(
                len(chirp),
                expected_length,
                f"Failed with params: fmin={fmin}, fmax={fmax}, duration={duration}, sr={sr}, linear={linear}",
            )

    def test_method_selection(self):
        """Test that the method parameter is correctly set based on the linear flag."""
        fmin = 100
        fmax = 1000
        duration = 1
        sr = 22050

        # We can't directly test the internal method parameter, but we can compare outputs
        linear_chirp = compute_chirp(fmin, fmax, duration, sr, True)
        log_chirp = compute_chirp(fmin, fmax, duration, sr, False)

        # The two chirps should be different due to different methods
        self.assertFalse(
            np.array_equal(linear_chirp, log_chirp),
            "Linear and logarithmic chirps should be different",
        )


if __name__ == "__main__":
    unittest.main()
