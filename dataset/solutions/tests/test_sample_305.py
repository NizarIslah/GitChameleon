import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_305 import compute_yin


class TestComputeYin(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.sr = 22050  # Sample rate in Hz
        self.fmin = 80  # Minimum frequency in Hz
        self.fmax = 500  # Maximum frequency in Hz
        self.frame_length = 2048
        self.trough_threshold = 0.1
        self.center = True
        self.pad_mode = "constant"
        self.win_length = None
        self.hop_length = None
        self.method = "parabolic"  # Interpolation method

    def test_default_parameters(self):
        """Test that compute_yin works with default parameters for win_length and hop_length."""
        # Create a simple audio signal
        duration = 0.5
        frequency = 220.0
        period = 1.0 / frequency
        phi = 0.0

        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * frequency * t + phi)

        # Compute YIN with default parameters
        f0 = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=duration,
            period=period,
            phi=phi,
            method=self.method,
            y=y,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=None,  # Default
            hop_length=None,  # Default
            trough_threshold=self.trough_threshold,
        )

        # Check that we got a valid result
        self.assertTrue(np.all(f0 > 0), "Expected positive frequency values")
        self.assertTrue(np.all(np.isfinite(f0)), "Expected finite frequency values")

    def test_different_center_and_pad_modes(self):
        """Test compute_yin with different center and pad_mode settings."""
        duration = 0.5
        frequency = 200.0
        period = 1.0 / frequency
        phi = 0.0

        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * frequency * t + phi)

        # Test with center=False
        f0_no_center = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=duration,
            period=period,
            phi=phi,
            method=self.method,
            y=y,
            frame_length=self.frame_length,
            center=False,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            trough_threshold=self.trough_threshold,
        )

        # Test with different pad_mode
        f0_reflect = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=duration,
            period=period,
            phi=phi,
            method=self.method,
            y=y,
            frame_length=self.frame_length,
            center=True,
            pad_mode="reflect",
            win_length=self.win_length,
            hop_length=self.hop_length,
            trough_threshold=self.trough_threshold,
        )

        # Check that we got valid results
        self.assertTrue(
            np.all(f0_no_center > 0),
            "Expected positive frequency values with center=False",
        )
        self.assertTrue(
            np.all(f0_reflect > 0),
            "Expected positive frequency values with pad_mode='reflect'",
        )

    def test_different_threshold(self):
        """Test compute_yin with different trough_threshold values."""
        duration = 0.5
        frequency = 180.0
        period = 1.0 / frequency
        phi = 0.0

        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * frequency * t + phi)

        # Test with a lower threshold
        f0_low_threshold = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=duration,
            period=period,
            phi=phi,
            method=self.method,
            y=y,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            trough_threshold=0.05,  # Lower threshold
        )

        # Test with a higher threshold
        f0_high_threshold = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=duration,
            period=period,
            phi=phi,
            method=self.method,
            y=y,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            trough_threshold=0.2,  # Higher threshold
        )

        # Check that we got valid results
        self.assertTrue(
            np.all(f0_low_threshold > 0),
            "Expected positive frequency values with low threshold",
        )
        self.assertTrue(
            np.all(f0_high_threshold > 0),
            "Expected positive frequency values with high threshold",
        )


if __name__ == "__main__":
    unittest.main()
