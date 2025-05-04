import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_278 import compute_rms


class TestComputeRMS(unittest.TestCase):
    def test_compute_rms_with_zeros(self):
        """Test that RMS of zeros is zero."""
        y = np.zeros(1024)
        rms = compute_rms(y)
        # RMS should return a 2D array with shape (1, n_frames)
        self.assertEqual(rms.ndim, 2)
        self.assertEqual(rms.shape[0], 1)
        # All values should be close to zero
        self.assertTrue(np.allclose(rms, 0.0))

    def test_compute_rms_with_ones(self):
        """Test RMS of array of ones."""
        y = np.ones(1024)
        rms = compute_rms(y)
        # RMS should return a 2D array with shape (1, n_frames)
        self.assertEqual(rms.ndim, 2)
        self.assertEqual(rms.shape[0], 1)
        # For an array of ones, RMS should be close to 1.0
        self.assertTrue(np.allclose(rms, 1.0))

    def test_compute_rms_with_sine_wave(self):
        """Test RMS of a sine wave."""
        # Create a sine wave with amplitude 1.0
        sr = 22050  # Sample rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        rms = compute_rms(y)

        # RMS should return a 2D array with shape (1, n_frames)
        self.assertEqual(rms.ndim, 2)
        self.assertEqual(rms.shape[0], 1)

        # For a sine wave with amplitude 1.0, the RMS should be close to 1/sqrt(2) = 0.7071...
        # However, due to windowing in librosa.feature.rms, we'll just check it's in a reasonable range
        self.assertTrue(np.all(rms > 0.5))
        self.assertTrue(np.all(rms < 0.8))

    def test_compute_rms_different_lengths(self):
        """Test RMS with arrays of different lengths."""
        # Test with a short array
        y_short = np.ones(512)
        rms_short = compute_rms(y_short)
        self.assertEqual(rms_short.ndim, 2)
        self.assertEqual(rms_short.shape[0], 1)

        # Test with a longer array
        y_long = np.ones(8192)
        rms_long = compute_rms(y_long)
        self.assertEqual(rms_long.ndim, 2)
        self.assertEqual(rms_long.shape[0], 1)

        # The number of frames should be different
        self.assertNotEqual(rms_short.shape[1], rms_long.shape[1])


if __name__ == "__main__":
    unittest.main()
