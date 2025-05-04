import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_277 import compute_rms


class TestComputeRMS(unittest.TestCase):
    def test_compute_rms_with_zeros(self):
        """Test that RMS of zeros is zero."""
        y = np.zeros(1024)
        rms = compute_rms(y)
        # librosa.feature.rmse returns a 2D array with shape (1, n_frames)
        self.assertEqual(rms.shape[0], 1)
        # Check that all values are close to zero
        self.assertTrue(np.allclose(rms, 0.0))

    def test_compute_rms_with_ones(self):
        """Test RMS of array of ones."""
        y = np.ones(1024)
        rms = compute_rms(y)
        # librosa.feature.rmse returns a 2D array with shape (1, n_frames)
        self.assertEqual(rms.shape[0], 1)
        # For an array of ones, the RMS should be close to 1.0
        self.assertTrue(np.all(rms > 0.9))

    def test_compute_rms_with_sine_wave(self):
        """Test RMS of a sine wave."""
        # Create a sine wave
        sr = 22050  # Sample rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        rms = compute_rms(y)

        # librosa.feature.rmse returns a 2D array with shape (1, n_frames)
        self.assertEqual(rms.shape[0], 1)

        # For a sine wave with amplitude 1, the RMS should be close to 1/sqrt(2) ≈ 0.707
        # But due to windowing effects in librosa.feature.rmse, we'll just check it's in a reasonable range
        self.assertTrue(np.all(rms > 0.5) and np.all(rms < 0.8))

    def test_compute_rms_shape(self):
        """Test the shape of the output from compute_rms."""
        # Test with different length inputs
        for length in [512, 1024, 2048]:
            y = np.random.random(length)
            rms = compute_rms(y)

            # librosa.feature.rmse returns a 2D array with shape (1, n_frames)
            self.assertEqual(rms.shape[0], 1)

            # The number of frames depends on the hop_length and n_fft parameters
            # which are default in the compute_rms function
            # We just check that the output has a reasonable shape
            self.assertTrue(rms.shape[1] > 0)


if __name__ == "__main__":
    unittest.main()
