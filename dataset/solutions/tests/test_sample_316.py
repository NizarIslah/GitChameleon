#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for sample_316.py
"""

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/samples"))
)
from sample_316 import compute_mfcc_to_mel


class TestComputeMfccToMel(unittest.TestCase):
    """Test cases for compute_mfcc_to_mel function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple audio signal for testing
        self.sr = 22050
        self.duration = 1.0  # 1 second
        self.y = np.sin(
            2
            * np.pi
            * 440
            * np.linspace(0, self.duration, int(self.sr * self.duration))
        )

        # Compute MFCCs from the audio signal
        self.n_mfcc = 13
        self.n_mels = 128
        self.mfcc = librosa.feature.mfcc(
            y=self.y, sr=self.sr, n_mfcc=self.n_mfcc, n_mels=self.n_mels
        )

    def test_compute_mfcc_to_mel_default_params(self):
        """Test compute_mfcc_to_mel with default parameters."""
        # Set the random seed to ensure reproducibility
        np.random.seed(0)

        # Call our function
        mel_power = compute_mfcc_to_mel(self.mfcc)

        # Call librosa's function directly with the same parameters
        np.random.seed(0)
        expected_mel_power = librosa.feature.inverse.mfcc_to_mel(self.mfcc)

        # Check that the output has the expected shape
        self.assertEqual(mel_power.shape, (self.n_mels, self.mfcc.shape[1]))

        # Check that our function returns the same result as librosa's function
        np.testing.assert_array_almost_equal(mel_power, expected_mel_power)

    def test_compute_mfcc_to_mel_custom_params(self):
        """Test compute_mfcc_to_mel with custom parameters."""
        # Set custom parameters
        n_mels = 64
        dct_type = 3
        norm = None
        ref = 0.1

        # Set the random seed to ensure reproducibility
        np.random.seed(0)

        # Call our function with custom parameters
        mel_power = compute_mfcc_to_mel(
            self.mfcc, n_mels=n_mels, dct_type=dct_type, norm=norm, ref=ref
        )

        # Call librosa's function directly with the same parameters
        np.random.seed(0)
        expected_mel_power = librosa.feature.inverse.mfcc_to_mel(self.mfcc)

        # Check that the output has the expected shape
        self.assertEqual(mel_power.shape, (self.n_mels, self.mfcc.shape[1]))

        # Check that our function returns the same result as librosa's function
        np.testing.assert_array_almost_equal(mel_power, expected_mel_power)

    def test_compute_mfcc_to_mel_with_random_mfcc(self):
        """Test compute_mfcc_to_mel with random MFCC data."""
        # Create random MFCC data
        np.random.seed(42)
        random_mfcc = np.random.rand(self.n_mfcc, 100)

        # Reset the random seed for the function call
        np.random.seed(0)

        # Call our function
        mel_power = compute_mfcc_to_mel(random_mfcc)

        # Check that the output has the expected shape
        self.assertEqual(mel_power.shape, (self.n_mels, random_mfcc.shape[1]))

        # Check that the output is not all zeros
        self.assertFalse(np.allclose(mel_power, 0))

        # Check that the output is non-negative (as it's a power spectrogram)
        self.assertTrue(np.all(mel_power >= 0))


if __name__ == "__main__":
    unittest.main()
