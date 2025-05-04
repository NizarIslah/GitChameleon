# Test file for sample_289.py
import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
)

from sample_289 import compute_fourier_tempogram


class TestFourierTempogram(unittest.TestCase):
    def test_compute_fourier_tempogram_shape(self):
        """Test that the output shape of the Fourier tempogram is correct."""
        # Create a simple onset envelope
        oenv = np.ones(100)
        sr = 22050
        hop_length = 512

        # Compute the Fourier tempogram
        tempogram = compute_fourier_tempogram(oenv, sr, hop_length)

        # Check the shape: should be (n_fft // 2 + 1, len(oenv) + 1)
        # n_fft is 384 as specified in the function
        expected_shape = (384 // 2 + 1, len(oenv) + 1)
        self.assertEqual(tempogram.shape, expected_shape)

    def test_compute_fourier_tempogram_dtype(self):
        """Test that the output dtype is complex."""
        oenv = np.random.random(50)
        sr = 22050
        hop_length = 512

        tempogram = compute_fourier_tempogram(oenv, sr, hop_length)

        # The output of STFT should be complex
        self.assertTrue(np.issubdtype(tempogram.dtype, np.complexfloating))

    def test_compute_fourier_tempogram_with_zeros(self):
        """Test the function with an all-zeros input."""
        oenv = np.zeros(80)
        sr = 22050
        hop_length = 512

        tempogram = compute_fourier_tempogram(oenv, sr, hop_length)

        # All values should be zero
        self.assertTrue(np.allclose(tempogram, 0))

    def test_compute_fourier_tempogram_with_sine(self):
        """Test the function with a sine wave input."""
        # Create a sine wave as the onset envelope
        t = np.linspace(0, 2 * np.pi, 100)
        oenv = np.sin(t)
        sr = 22050
        hop_length = 512

        tempogram = compute_fourier_tempogram(oenv, sr, hop_length)

        # Check that the output is not all zeros
        self.assertFalse(np.allclose(tempogram, 0))

        # The shape should be consistent
        expected_shape = (384 // 2 + 1, len(oenv) + 1)
        self.assertEqual(tempogram.shape, expected_shape)

    def test_compute_fourier_tempogram_parameters(self):
        """Test that the function uses the correct STFT parameters."""
        # Create a simple onset envelope
        oenv = np.random.random(60)
        sr = 22050
        hop_length = 512

        # Compute the Fourier tempogram
        tempogram = compute_fourier_tempogram(oenv, sr, hop_length)

        # Manually compute STFT with the same parameters for comparison
        from librosa.core.spectrum import stft


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

sol = compute_fourier_tempogram(oenv, sr, hop_length)
test_sol = stft(oenv, n_fft=384, hop_length=1, center=True, window="hann")
assert np.array_equal(test_sol, sol)
