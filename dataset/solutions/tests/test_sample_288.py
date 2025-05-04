import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_288 import compute_lpc_coef


class TestLPCCoefficients(unittest.TestCase):
    def setUp(self):
        # Create a simple sine wave as test audio signal
        self.sr = 22050  # Sample rate
        self.duration = 0.5  # Duration in seconds
        self.t = np.linspace(
            0, self.duration, int(self.sr * self.duration), endpoint=False
        )
        self.y = np.sin(2 * np.pi * 440 * self.t)  # 440 Hz sine wave

    def test_compute_lpc_coef_returns_correct_shape(self):
        """Test that the function returns coefficients of the expected shape."""
        order = 10
        coeffs = compute_lpc_coef(self.y, self.sr, order)

        # LPC coefficients should be of length order+1
        self.assertEqual(len(coeffs), order + 1)

    def test_compute_lpc_coef_first_coefficient_is_one(self):
        """Test that the first coefficient is always 1."""
        order = 10
        coeffs = compute_lpc_coef(self.y, self.sr, order)

        # The first coefficient should always be 1
        self.assertEqual(coeffs[0], 1.0)

    def test_compute_lpc_coef_matches_librosa_direct(self):
        """Test that our wrapper function matches direct librosa call."""
        order = 10
        our_coeffs = compute_lpc_coef(self.y, self.sr, order)
        librosa_coeffs = librosa.lpc(self.y, order)

        # The coefficients should be identical
        np.testing.assert_array_almost_equal(our_coeffs, librosa_coeffs)

    def test_compute_lpc_coef_with_different_orders(self):
        """Test the function with different filter orders."""
        for order in [5, 12, 20]:
            coeffs = compute_lpc_coef(self.y, self.sr, order)
            self.assertEqual(len(coeffs), order + 1)
            self.assertEqual(coeffs[0], 1.0)

    def test_compute_lpc_coef_with_real_audio(self):
        """Test the function with a more complex audio signal."""
        # Create a more complex signal with multiple frequencies
        t = np.linspace(0, 1.0, int(self.sr * 1.0), endpoint=False)
        y = (
            0.5 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.random.randn(len(t))
        )

        order = 15
        coeffs = compute_lpc_coef(y, self.sr, order)

        # Check basic properties
        self.assertEqual(len(coeffs), order + 1)
        self.assertEqual(coeffs[0], 1.0)

        # The coefficients should capture the spectral characteristics
        # We can't easily predict exact values, but we can check they're non-zero
        self.assertTrue(np.any(np.abs(coeffs[1:]) > 1e-5))


if __name__ == "__main__":
    unittest.main()
