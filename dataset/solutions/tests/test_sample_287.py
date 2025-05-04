#!/usr/bin/env python3
# Test file for sample_287.py

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_287 import compute_lpc_coef


class TestComputeLPCCoef(unittest.TestCase):
    """Test cases for the compute_lpc_coef function."""

    def test_basic_functionality(self):
        """Test that the function works with a simple sine wave."""
        # Create a simple sine wave
        sr = 22050  # Sample rate in Hz
        duration = 0.1  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Test with different orders
        for order in [5, 10, 15]:
            coeffs = compute_lpc_coef(y, sr, order)

            # Check that the output has the expected shape
            self.assertEqual(len(coeffs), order + 1)

            # Check that the first coefficient is 1
            self.assertEqual(coeffs[0], 1.0)

            # Check that the coefficients are finite
            self.assertTrue(np.all(np.isfinite(coeffs)))

    def test_error_handling(self):
        """Test that the function raises appropriate errors."""
        # Create a signal that will cause numerical issues
        y = np.zeros(100, dtype=np.float32)  # All zeros will cause division by zero
        sr = 22050
        order = 5

        # The function should raise a FloatingPointError
        with self.assertRaises(FloatingPointError):
            compute_lpc_coef(y, sr, order)

    def test_with_real_audio_simulation(self):
        """Test with a more complex signal that simulates real audio."""
        # Create a more complex signal (sum of sine waves)
        sr = 22050
        duration = 0.2
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Create a signal with multiple frequency components
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        y += 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz (first harmonic)
        y += 0.1 * np.sin(2 * np.pi * 1320 * t)  # 1320 Hz (second harmonic)
        y += 0.05 * np.random.randn(len(t))  # Add some noise

        order = 20
        coeffs = compute_lpc_coef(y, sr, order)

        # Check basic properties
        self.assertEqual(len(coeffs), order + 1)
        self.assertEqual(coeffs[0], 1.0)
        self.assertTrue(np.all(np.isfinite(coeffs)))

        # For LPC coefficients of a signal with strong periodicity,
        # we expect some of the coefficients to have larger magnitudes
        self.assertTrue(np.max(np.abs(coeffs[1:])) > 0.1)


if __name__ == "__main__":
    unittest.main()
