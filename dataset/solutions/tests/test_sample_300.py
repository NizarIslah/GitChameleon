# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_300 import compute_chirp


class TestComputeChirp(unittest.TestCase):
    def test_compute_chirp_basic(self):
        """Test that compute_chirp returns a numpy array with expected length."""
        fmin = 100
        fmax = 1000
        duration = 2
        sr = 22050
        linear = True

        chirp_signal = compute_chirp(fmin, fmax, duration, sr, linear)

        # Check that the output is a numpy array
        self.assertIsInstance(chirp_signal, np.ndarray)

        # Check that the length of the signal matches the expected duration
        expected_length = duration * sr
        self.assertEqual(len(chirp_signal), expected_length)

    def test_compute_chirp_different_params(self):
        """Test compute_chirp with different parameter values."""
        fmin = 200
        fmax = 2000
        duration = 1
        sr = 44100
        linear = False

        chirp_signal = compute_chirp(fmin, fmax, duration, sr, linear)

        # Check that the output is a numpy array
        self.assertIsInstance(chirp_signal, np.ndarray)

        # Check that the length of the signal matches the expected duration
        expected_length = duration * sr
        self.assertEqual(len(chirp_signal), expected_length)

    def test_compute_chirp_frequency_range(self):
        """Test that the chirp signal contains frequencies in the expected range."""
        fmin = 100
        fmax = 1000
        duration = 5
        sr = 22050
        linear = True

        chirp_signal = compute_chirp(fmin, fmax, duration, sr, linear)

        # Check that the signal is not all zeros
        self.assertFalse(np.all(chirp_signal == 0))

        # Check that the signal has some variation (not a constant value)
        self.assertTrue(np.std(chirp_signal) > 0)


if __name__ == "__main__":
    unittest.main()
