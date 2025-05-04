# Test file for sample_290.py
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_290 import compute_fourier_tempogram


class TestFourierTempogram(unittest.TestCase):
    """Test cases for the compute_fourier_tempogram function."""

    def setUp(self):
        """Set up test data."""
        # Create a simple onset envelope for testing
        # Using a sine wave as a simple test case
        self.sr = 22050  # Standard sample rate in Hz
        self.hop_length = 512  # Standard hop length

        # Create a simple onset envelope (10 frames of data)
        t = np.linspace(0, 1, 10)
        self.oenv = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    def test_output_shape(self):
        """Test that the output has the expected shape."""
        tempogram = compute_fourier_tempogram(self.oenv, self.sr, self.hop_length)

        # The output should be a 2D array
        self.assertEqual(len(tempogram.shape), 2)

        # First dimension should be related to the number of frequency bins
        # Second dimension should match the expected output length
        self.assertEqual(tempogram.shape[1], len(self.oenv) + 1)

    def test_output_type(self):
        """Test that the output has the expected data type."""
        tempogram = compute_fourier_tempogram(self.oenv, self.sr, self.hop_length)

        # Output should be a complex-valued numpy array
        self.assertTrue(np.issubdtype(tempogram.dtype, np.complexfloating))

    def test_parameter_passing(self):
        """Test that parameters are correctly passed to the librosa function."""
        # We'll use a mock to verify this in a real-world scenario,
        # but for this test we'll just verify the function doesn't raise exceptions
        # with different parameter values

        # Test with different sr values
        tempogram1 = compute_fourier_tempogram(self.oenv, 11025, self.hop_length)
        tempogram2 = compute_fourier_tempogram(self.oenv, 44100, self.hop_length)

        # The shapes should be the same since we're using the same oenv
        self.assertEqual(tempogram1.shape, tempogram2.shape)

        # Test with different hop_length values
        tempogram3 = compute_fourier_tempogram(self.oenv, self.sr, 256)
        tempogram4 = compute_fourier_tempogram(self.oenv, self.sr, 1024)

        # The shapes should be the same since we're using the same oenv
        self.assertEqual(tempogram3.shape, tempogram4.shape)

    def test_with_zeros(self):
        """Test with an onset envelope of all zeros."""
        zero_oenv = np.zeros(10)
        tempogram = compute_fourier_tempogram(zero_oenv, self.sr, self.hop_length)

        # The output should have the same shape as with non-zero input
        self.assertEqual(tempogram.shape[1], len(zero_oenv) + 1)

        # All values should be close to zero
        self.assertTrue(np.allclose(np.abs(tempogram), 0))


if __name__ == "__main__":
    unittest.main()
