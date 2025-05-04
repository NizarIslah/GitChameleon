#!/usr/bin/env python
# Test file for sample_311.py

import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sample_311 import compute_griffinlim_cqt


class TestGriffinLimCQT(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.sr = 22050  # Sample rate
        self.hop_length = 512  # Hop length
        self.bins_per_octave = 12  # Bins per octave
        self.n_bins = 84  # Number of frequency bins
        self.fmin = librosa.note_to_hz("C1")  # Minimum frequency

        # Create a simple audio signal (sine wave)
        duration = 3  # seconds
        t = np.linspace(0, duration, int(duration * self.sr), endpoint=False)
        self.y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Compute CQT of the signal
        self.C = librosa.cqt(
            self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

    def test_basic_functionality(self):
        """Test the basic functionality of the Griffin-Lim CQT algorithm."""
        # Call the function with minimal parameters
        y_reconstructed = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=5,  # Use a small number of iterations for testing
            hop_length=self.hop_length,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.0,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=None,
            momentum=0.99,
            init=None,
        )

        # Check that the output is a numpy array
        self.assertIsInstance(y_reconstructed, np.ndarray)

        # Check that the output has the expected shape (should be similar to input)
        # The length might not be exactly the same due to windowing effects
        self.assertEqual(y_reconstructed.ndim, 1)

        # Check that the output is not all zeros
        self.assertGreater(np.abs(y_reconstructed).sum(), 0)

    def test_random_init(self):
        """Test the function with random initialization."""
        y_reconstructed = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=3,  # Use a small number of iterations for testing
            hop_length=self.hop_length,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.0,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=None,
            momentum=0.99,
            init="random",
        )

        # Check that the output is a numpy array
        self.assertIsInstance(y_reconstructed, np.ndarray)

        # Check that the output has the expected shape
        self.assertEqual(y_reconstructed.ndim, 1)

        # Check that the output is not all zeros
        self.assertGreater(np.abs(y_reconstructed).sum(), 0)

    def test_different_momentum(self):
        """Test the function with different momentum values."""
        y_reconstructed = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=3,  # Use a small number of iterations for testing
            hop_length=self.hop_length,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.0,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=None,
            momentum=0.5,  # Different momentum value
            init=None,
        )

        # Check that the output is a numpy array
        self.assertIsInstance(y_reconstructed, np.ndarray)

        # Check that the output is not all zeros
        self.assertGreater(np.abs(y_reconstructed).sum(), 0)

    def test_with_length_parameter(self):
        """Test the function with a specific length parameter."""
        target_length = len(self.y)
        y_reconstructed = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=3,  # Use a small number of iterations for testing
            hop_length=self.hop_length,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.0,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=target_length,  # Specify the exact length
            momentum=0.99,
            init=None,
        )

        # Check that the output has the expected length
        self.assertEqual(len(y_reconstructed), target_length)

    def test_with_none_fmin(self):
        """Test the function with fmin=None."""
        y_reconstructed = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=3,  # Use a small number of iterations for testing
            hop_length=self.hop_length,
            fmin=None,  # Test with None to trigger the default value
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.0,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=None,
            momentum=0.99,
            init=None,
        )

        # Check that the output is a numpy array
        self.assertIsInstance(y_reconstructed, np.ndarray)

        # Check that the output is not all zeros
        self.assertGreater(np.abs(y_reconstructed).sum(), 0)


if __name__ == "__main__":
    unittest.main()
