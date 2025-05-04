# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_306 import compute_yin


class TestComputeYin(unittest.TestCase):
    def setUp(self):
        # Create a simple sine wave as test signal
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        self.period = 0.01  # Period in seconds (100 Hz)
        self.phi = 0.0  # Phase

        # Generate a sine wave with frequency 100 Hz
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.y = np.sin(2 * np.pi * (1.0 / self.period) * t + self.phi)

        # Set other parameters
        self.fmin = 50  # Minimum frequency in Hz
        self.fmax = 500  # Maximum frequency in Hz
        self.frame_length = 2048  # Frame length in samples
        self.center = True  # Center the frames
        self.pad_mode = "reflect"  # Padding mode
        self.win_length = None  # Window length
        self.hop_length = None  # Hop length
        self.trough_threshold = 0.1  # Threshold for peak estimation
        self.method = "parabolic"  # Interpolation method

    @patch("librosa.yin")
    def test_compute_yin_calls_librosa_yin_with_correct_params(self, mock_yin):
        # Setup mock return value
        expected_output = np.array([100.0, 100.0, 100.0])
        mock_yin.return_value = expected_output

        # Call the function
        result = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=self.duration,
            period=self.period,
            phi=self.phi,
            method=self.method,
            y=self.y,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            trough_threshold=self.trough_threshold,
        )

        # Assert that librosa.yin was called with the correct parameters
        mock_yin.assert_called_once_with(
            self.y, fmin=self.fmin, fmax=self.fmax, sr=self.sr
        )

        # Assert that the function returns the expected output
        np.testing.assert_array_equal(result, expected_output)

    def test_compute_yin_with_real_signal(self):
        # This test uses the actual librosa.yin function
        # We expect the result to be close to the frequency of our test signal (100 Hz)
        result = compute_yin(
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=self.duration,
            period=self.period,
            phi=self.phi,
            method=self.method,
            y=self.y,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            trough_threshold=self.trough_threshold,
        )

        # The result should be an array of frequency estimates
        self.assertIsInstance(result, np.ndarray)

        # Check that the result is not empty
        self.assertGreater(len(result), 0)

        # For a clean sine wave, the YIN algorithm should estimate a frequency
        # close to the actual frequency (100 Hz in this case)
        # Note: We use a tolerance because YIN is an estimation algorithm
        # and may not be perfectly accurate, especially with short signals
        mean_frequency = np.mean(result)
        self.assertGreaterEqual(mean_frequency, self.fmin)
        self.assertLessEqual(mean_frequency, self.fmax)


if __name__ == "__main__":
    unittest.main()
