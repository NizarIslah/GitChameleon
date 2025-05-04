#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for sample_293.py
"""

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_293 import compute_times_like


class TestComputeTimesLike(unittest.TestCase):
    """Test cases for compute_times_like function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample audio signal
        self.sr = 22050  # Sample rate in Hz
        self.hop_length = 512  # Hop length in samples
        self.y = np.random.random(self.sr * 3)  # 3 seconds of random audio

    def test_scalar_input(self):
        """Test compute_times_like with scalar input."""
        # Test with a scalar value for D (e.g., number of frames)
        num_frames = 100
        times = compute_times_like(self.y, self.sr, self.hop_length, num_frames)

        # Expected: frames converted to time in seconds
        expected_times = np.arange(num_frames) * self.hop_length / float(self.sr)

        # Check shape and values
        self.assertEqual(times.shape, (num_frames,))
        np.testing.assert_allclose(times, expected_times)

        # Check data type
        self.assertTrue(np.issubdtype(times.dtype, np.floating))

    def test_array_input(self):
        """Test compute_times_like with array input."""
        # Create a mock spectrogram with 10 frequency bins and 50 time frames
        mock_spectrogram = np.random.random((10, 50))

        times = compute_times_like(self.y, self.sr, self.hop_length, mock_spectrogram)

        # Expected: frames converted to time in seconds
        expected_times = (
            np.arange(mock_spectrogram.shape[-1]) * self.hop_length / float(self.sr)
        )

        # Check shape and values
        self.assertEqual(times.shape, (mock_spectrogram.shape[-1],))
        np.testing.assert_allclose(times, expected_times)

        # Check data type
        self.assertTrue(np.issubdtype(times.dtype, np.floating))

    def test_empty_spectrogram(self):
        """Test compute_times_like with an empty spectrogram."""
        # Create an empty spectrogram with 0 time frames
        mock_spectrogram = np.random.random((10, 0))

        times = compute_times_like(self.y, self.sr, self.hop_length, mock_spectrogram)

        # Should return an empty array
        self.assertEqual(times.shape, (0,))

    def test_different_sample_rates(self):
        """Test compute_times_like with different sample rates."""
        num_frames = 100

        # Test with different sample rates
        for test_sr in [8000, 16000, 44100, 48000]:
            times = compute_times_like(self.y, test_sr, self.hop_length, num_frames)
            expected_times = np.arange(num_frames) * self.hop_length / float(test_sr)
            np.testing.assert_allclose(times, expected_times)

    def test_different_hop_lengths(self):
        """Test compute_times_like with different hop lengths."""
        num_frames = 100

        # Test with different hop lengths
        for test_hop in [256, 512, 1024, 2048]:
            times = compute_times_like(self.y, self.sr, test_hop, num_frames)
            expected_times = np.arange(num_frames) * test_hop / float(self.sr)
            np.testing.assert_allclose(times, expected_times)


if __name__ == "__main__":
    unittest.main()
