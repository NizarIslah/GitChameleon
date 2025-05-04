import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest
from typing import Optional, Union

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_313 import compute_mel_to_audio


class TestMelToAudio(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(0)

        # Create test parameters
        self.sr = 22050  # Sample rate
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length
        self.n_mels = 128  # Number of mel bands

        # Generate a simple test signal (sine wave)
        duration = 1.0  # seconds
        self.y = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(duration * self.sr))
        )

        # Compute STFT
        self.S = np.abs(
            librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length)
        )

        # Compute mel spectrogram
        self.M = librosa.feature.melspectrogram(
            S=self.S**2,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def test_compute_mel_to_audio_shape(self):
        """Test that the output has the expected shape."""
        # Run the function with minimal iterations for speed
        n_iter = 2
        y_reconstructed = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=n_iter,
            length=None,
            dtype=np.float32,
        )

        # Check that the output is a numpy array
        self.assertIsInstance(y_reconstructed, np.ndarray)

        # Check that the output is 1D
        self.assertEqual(y_reconstructed.ndim, 1)

        # Check that the output has a reasonable length
        # The reconstructed audio might not be exactly the same length as the input
        # due to windowing effects, but it should be close
        self.assertGreater(len(y_reconstructed), 0)

    def test_compute_mel_to_audio_dtype(self):
        """Test that the output has the expected dtype."""
        # Test with float32
        y_reconstructed_f32 = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=1,
            length=None,
            dtype=np.float32,
        )

        self.assertEqual(y_reconstructed_f32.dtype, np.float32)

        # Test with float64
        y_reconstructed_f64 = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=1,
            length=None,
            dtype=np.float64,
        )

        self.assertEqual(y_reconstructed_f64.dtype, np.float64)

    def test_compute_mel_to_audio_deterministic(self):
        """Test that the function is deterministic with fixed random seed."""
        # Run the function twice with the same parameters
        np.random.seed(0)  # Reset seed
        y_reconstructed1 = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=1,
            length=None,
            dtype=np.float32,
        )

        np.random.seed(0)  # Reset seed
        y_reconstructed2 = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=1,
            length=None,
            dtype=np.float32,
        )

        # Check that the outputs are identical
        np.testing.assert_array_equal(y_reconstructed1, y_reconstructed2)

    def test_compute_mel_to_audio_different_parameters(self):
        """Test that the function works with different parameters."""
        # Test with different window length
        win_length = 1024
        y_reconstructed = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=1,
            length=None,
            dtype=np.float32,
        )

        self.assertIsInstance(y_reconstructed, np.ndarray)
        self.assertEqual(y_reconstructed.ndim, 1)

        # Test with different window type
        y_reconstructed = compute_mel_to_audio(
            y=self.y,
            sr=self.sr,
            S=self.S,
            M=self.M,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hamming",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=1,
            length=None,
            dtype=np.float32,
        )

        self.assertIsInstance(y_reconstructed, np.ndarray)
        self.assertEqual(y_reconstructed.ndim, 1)


if __name__ == "__main__":
    unittest.main()
