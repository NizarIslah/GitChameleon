#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for sample_315.py which contains the compute_mfcc_to_mel function.
"""

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np
import scipy

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/samples"))
)
from sample_315 import compute_mfcc_to_mel


class TestComputeMfccToMel(unittest.TestCase):
    """Test cases for the compute_mfcc_to_mel function."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(0)

        # Create a sample MFCC matrix (20 coefficients, 10 frames)
        self.mfcc_sample = np.random.rand(20, 10)

        # Default parameters
        self.n_mels_default = 128
        self.dct_type_default = 2
        self.norm_default = "ortho"
        self.ref_default = 1.0

    def test_output_shape(self):
        """Test that the output shape is correct."""
        # Test with default parameters
        mel_spec = compute_mfcc_to_mel(self.mfcc_sample)
        self.assertEqual(mel_spec.shape, (self.n_mels_default, 10))

        # Test with custom n_mels
        n_mels_custom = 64
        mel_spec = compute_mfcc_to_mel(self.mfcc_sample, n_mels=n_mels_custom)
        self.assertEqual(mel_spec.shape, (n_mels_custom, 10))

    def test_output_values_positive(self):
        """Test that the output values are positive (as expected for power spectrogram)."""
        mel_spec = compute_mfcc_to_mel(self.mfcc_sample)
        self.assertTrue(
            np.all(mel_spec >= 0),
            "Mel spectrogram should contain only non-negative values",
        )

    def test_different_norm_values(self):
        """Test with different normalization values."""
        # Test with 'ortho' normalization (default)
        mel_spec_ortho = compute_mfcc_to_mel(self.mfcc_sample, norm="ortho")
        # Test with None normalization
        mel_spec_none = compute_mfcc_to_mel(self.mfcc_sample, norm=None)

        # The outputs should be different for different normalization
        self.assertFalse(np.allclose(mel_spec_ortho, mel_spec_none))

    def test_different_ref_values(self):
        """Test with different reference values."""
        # Test with default ref=1.0
        mel_spec_ref1 = compute_mfcc_to_mel(self.mfcc_sample, ref=1.0)
        # Test with ref=2.0
        mel_spec_ref2 = compute_mfcc_to_mel(self.mfcc_sample, ref=2.0)

        # The outputs should be different for different reference values
        self.assertFalse(np.allclose(mel_spec_ref1, mel_spec_ref2))

    def test_implementation_correctness(self):
        """Test that our implementation matches the expected behavior."""
        # Create a simple test case
        test_mfcc = np.array([[1.0, 2.0], [3.0, 4.0]])
        n_mels = 4

        # Compute the expected result manually
        np.random.seed(0)  # Match the seed in the function
        expected_logmel = scipy.fftpack.idct(
            test_mfcc, axis=0, type=2, norm="ortho", n=n_mels
        )
        expected_result = librosa.db_to_power(expected_logmel, ref=1.0)

        # Compute the actual result
        np.random.seed(0)  # Reset seed to ensure consistency
        actual_result = compute_mfcc_to_mel(test_mfcc, n_mels=n_mels)

        # Check that the results match
        self.assertTrue(np.allclose(actual_result, expected_result))


if __name__ == "__main__":
    unittest.main()
