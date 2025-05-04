#!/usr/bin/env python3
# Test file for sample_295.py

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_295 import compute_samples_like


class TestComputeSamplesLike(unittest.TestCase):
    """Test cases for the compute_samples_like function."""

    def test_scalar_input(self):
        """Test compute_samples_like with scalar input."""
        # Create test inputs
        y = np.zeros(1000)  # Dummy audio signal
        sr = 22050  # Standard sampling rate
        D = 10  # Scalar spectrogram length
        hop_length = 512  # Standard hop length

        # Call the function
        samples = compute_samples_like(y, sr, D, hop_length)

        # Verify the output
        expected = np.array([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608])
        np.testing.assert_array_equal(samples, expected)
        self.assertEqual(len(samples), D)
        self.assertEqual(samples.dtype, np.int64)

    def test_array_input(self):
        """Test compute_samples_like with array input."""
        # Create test inputs
        y = np.zeros(1000)  # Dummy audio signal
        sr = 22050  # Standard sampling rate
        # Create a dummy spectrogram with 5 frames
        D = np.zeros((128, 5))  # 128 frequency bins, 5 time frames
        hop_length = 256  # Different hop length

        # Call the function
        samples = compute_samples_like(y, sr, D, hop_length)

        # Verify the output
        expected = np.array([0, 256, 512, 768, 1024])
        np.testing.assert_array_equal(samples, expected)
        self.assertEqual(len(samples), D.shape[1])
        self.assertEqual(samples.dtype, np.int64)

    def test_empty_array(self):
        """Test compute_samples_like with an empty array."""
        # Create test inputs
        y = np.zeros(1000)  # Dummy audio signal
        sr = 22050  # Standard sampling rate
        # Create an empty spectrogram with 0 frames
        D = np.zeros((128, 0))  # 128 frequency bins, 0 time frames
        hop_length = 512  # Standard hop length

        # Call the function
        samples = compute_samples_like(y, sr, D, hop_length)

        # Verify the output
        expected = np.array([], dtype=int)
        np.testing.assert_array_equal(samples, expected)
        self.assertEqual(len(samples), 0)
        self.assertEqual(samples.dtype, np.int64)

    def test_unused_parameters(self):
        """Test that y and sr parameters don't affect the output."""
        # Create two different sets of inputs that only differ in y and sr
        y1 = np.zeros(1000)
        sr1 = 22050
        y2 = np.ones(500)
        sr2 = 44100
        D = 5
        hop_length = 128

        # Call the function with both sets
        samples1 = compute_samples_like(y1, sr1, D, hop_length)
        samples2 = compute_samples_like(y2, sr2, D, hop_length)

        # Verify that the outputs are identical
        np.testing.assert_array_equal(samples1, samples2)


if __name__ == "__main__":
    unittest.main()
