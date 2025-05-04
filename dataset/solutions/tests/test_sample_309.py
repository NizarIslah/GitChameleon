# Test file for sample_309.py
import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_309 import compute_vqt


class TestComputeVQT(unittest.TestCase):
    def setUp(self):
        # Create a simple sine wave as test audio
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        self.y = librosa.tone(440, sr=self.sr, duration=self.duration)  # A440 sine wave

    def test_compute_vqt_basic(self):
        """Test that compute_vqt runs without errors with basic parameters."""
        hop_length = 512
        fmin = 32.7  # C1 frequency
        n_bins = 84
        bins_per_octave = 12

        # Call the function with minimal required parameters
        V = compute_vqt(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            gamma=0,
            bins_per_octave=bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type=None,
            dtype=None,
        )

        # Check that the output has the expected shape
        # The number of frequency bins should match n_bins
        self.assertEqual(V.shape[0], n_bins)

        # The number of time frames should be related to the signal length and hop_length
        expected_frames = 1 + (len(self.y) - 1) // hop_length
        self.assertLessEqual(V.shape[1], expected_frames)

        # Check that the output is complex-valued
        self.assertTrue(np.iscomplexobj(V))

    def test_compute_vqt_different_bins(self):
        """Test compute_vqt with different number of bins."""
        hop_length = 512
        fmin = 32.7  # C1 frequency
        n_bins = 48  # Fewer bins
        bins_per_octave = 12

        V = compute_vqt(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            gamma=0,
            bins_per_octave=bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type=None,
            dtype=None,
        )

        # Check that the output has the expected shape
        self.assertEqual(V.shape[0], n_bins)

    def test_compute_vqt_with_gamma(self):
        """Test compute_vqt with non-zero gamma (for variable-Q transform)."""
        hop_length = 512
        fmin = 32.7  # C1 frequency
        n_bins = 84
        bins_per_octave = 12
        gamma = 25  # Non-zero gamma for variable-Q transform

        V = compute_vqt(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            gamma=gamma,
            bins_per_octave=bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type=None,
            dtype=None,
        )

        # Check that the output has the expected shape
        self.assertEqual(V.shape[0], n_bins)

    def test_compute_vqt_with_explicit_dtype(self):
        """Test compute_vqt with explicit dtype specification."""
        hop_length = 512
        fmin = 32.7  # C1 frequency
        n_bins = 84
        bins_per_octave = 12

        V = compute_vqt(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            gamma=0,
            bins_per_octave=bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type=None,
            dtype=np.complex64,
        )

        # Check that the output has the expected dtype
        self.assertEqual(V.dtype, np.complex64)

    def test_compute_vqt_with_different_hop_length(self):
        """Test compute_vqt with different hop length."""
        hop_length = 1024  # Larger hop length
        fmin = 32.7  # C1 frequency
        n_bins = 84
        bins_per_octave = 12

        V = compute_vqt(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            gamma=0,
            bins_per_octave=bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type=None,
            dtype=None,
        )

        # Check that the output has the expected shape
        self.assertEqual(V.shape[0], n_bins)

        # The number of time frames should be related to the signal length and hop_length
        expected_frames = 1 + (len(self.y) - 1) // hop_length
        self.assertLessEqual(V.shape[1], expected_frames)


if __name__ == "__main__":
    unittest.main()
