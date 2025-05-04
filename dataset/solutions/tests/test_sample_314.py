# Add the parent directory to import sys
import os
import sys
import unittest

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_314 import compute_mel_to_audio


class TestSample314(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility
        np.random.seed(0)

        # Create sample data for testing
        self.sr = 22050  # Sample rate
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length
        self.n_mels = 128  # Number of mel bands

        # Generate a simple audio signal (sine wave)
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        self.y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Compute mel spectrogram
        self.S = librosa.feature.melspectrogram(
            y=self.y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Other parameters needed for the function
        self.win_length = None
        self.window = "hann"
        self.center = True
        self.pad_mode = "reflect"
        self.power = 2.0
        self.n_iter = 32
        self.length = None
        self.dtype = np.float32

    def test_compute_mel_to_audio(self):
        """Test that compute_mel_to_audio returns the expected output."""
        # Call the function
        result = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=None,  # Not used in the function
            M=self.S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            power=self.power,
            n_iter=self.n_iter,
            length=self.length,
            dtype=self.dtype,
        )

        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Verify the result has the expected shape (should be similar to original audio length)
        expected_length = len(self.y)
        self.assertGreaterEqual(
            len(result), expected_length * 0.5
        )  # Allow some flexibility in length

        # Verify direct call to librosa function produces same result
        np.random.seed(0)  # Reset seed to match the function
        expected = librosa.feature.inverse.mel_to_audio(self.S)
        np.testing.assert_array_equal(result, expected)

    def test_seed_consistency(self):
        """Test that the function produces consistent results with fixed seed."""
        # Call the function twice with the same inputs
        result1 = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=None,
            M=self.S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            power=self.power,
            n_iter=self.n_iter,
            length=self.length,
            dtype=self.dtype,
        )

        result2 = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=None,
            M=self.S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            power=self.power,
            n_iter=self.n_iter,
            length=self.length,
            dtype=self.dtype,
        )

        # Results should be identical due to fixed seed
        np.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
    unittest.main()
