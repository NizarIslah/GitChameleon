import os
import sys
import unittest
from typing import Optional

import numpy as np

# Add the parent directory to the path so we can import the sample module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_285 import compute_griffinlim


class TestGriffinLim(unittest.TestCase):
    def setUp(self):
        # Create a simple spectrogram for testing
        self.n_fft = 512
        self.hop_length = 128
        self.sr = 22050
        self.duration = 1.0  # 1 second
        
        # Generate a simple sine wave as test signal
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create a magnitude spectrogram
        S_complex = np.abs(np.fft.rfft(self.y[:self.n_fft]))
        # Reshape to match expected STFT shape (n_freq, n_frames)
        n_frames = int(np.ceil(len(self.y) / self.hop_length))
        self.S = np.tile(S_complex[:, np.newaxis], (1, n_frames))

    def test_compute_griffinlim_basic(self):
        """Test that compute_griffinlim runs without errors with basic parameters."""
        result = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=42,
            n_iter=5,  # Use fewer iterations for a quicker test
            hop_length=self.hop_length,
            win_length=None,
            window='hann',
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode='reflect',
            n_fft=self.n_fft
        )
        
        # Check that the result is a numpy array with the expected shape
        self.assertIsInstance(result, np.ndarray)
        # The result should be approximately the same length as the input
        self.assertGreaterEqual(len(result), len(self.y) * 0.9)
        self.assertLessEqual(len(result), len(self.y) * 1.1)

    def test_compute_griffinlim_different_parameters(self):
        """Test compute_griffinlim with different parameter values."""
        result = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=42,
            n_iter=3,  # Use fewer iterations for testing
            hop_length=self.hop_length,
            win_length=self.n_fft // 2,  # Explicit win_length
            window='hamming',  # Different window
            center=False,  # Different centering
            dtype=np.float64,  # Different dtype
            length=None,
            pad_mode='constant',  # Different pad_mode
            n_fft=self.n_fft
        )
        
        # Check that the result is a numpy array with the expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_compute_griffinlim_reproducibility(self):
        """Test that compute_griffinlim produces the same output with the same random_state."""
        # Run the function twice with the same random_state
        result1 = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=42,
            n_iter=2,
            hop_length=self.hop_length,
            win_length=None,
            window='hann',
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode='reflect',
            n_fft=self.n_fft
        )
        
        result2 = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=42,
            n_iter=2,
            hop_length=self.hop_length,
            win_length=None,
            window='hann',
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode='reflect',
            n_fft=self.n_fft
        )
        
        # The results should be identical with the same random_state
        np.testing.assert_array_equal(result1, result2)
        
        # Run with a different random_state
        result3 = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=43,  # Different random_state
            n_iter=2,
            hop_length=self.hop_length,
            win_length=None,
            window='hann',
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode='reflect',
            n_fft=self.n_fft
        )
        
        # The results should differ with a different random_state
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(result1, result3)


if __name__ == '__main__':
    unittest.main()