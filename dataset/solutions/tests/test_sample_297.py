import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_297 import compute_tone


class TestComputeTone(unittest.TestCase):
    def test_output_length(self):
        """Test that the output has the correct length."""
        frequency = 440  # A4 note
        sr = 22050  # Common sampling rate
        length = 1000

        tone = compute_tone(frequency, sr, length)

        self.assertEqual(len(tone), length)

    def test_output_type(self):
        """Test that the output is a numpy array."""
        frequency = 440
        sr = 22050
        length = 1000

        tone = compute_tone(frequency, sr, length)

        self.assertIsInstance(tone, np.ndarray)

    def test_output_range(self):
        """Test that the output values are in the correct range for a cosine wave."""
        frequency = 440
        sr = 22050
        length = 1000

        tone = compute_tone(frequency, sr, length)

        # Cosine values should be between -1 and 1
        self.assertTrue(np.all(tone >= -1.0))
        self.assertTrue(np.all(tone <= 1.0))

    def test_phase_shift(self):
        """Test that the phase shift is correctly applied."""
        frequency = 440
        sr = 22050
        length = 1000

        tone = compute_tone(frequency, sr, length)

        # With phi = -np.pi * 0.5, the first value should be close to 0
        # (cosine of -pi/2 is 0)
        self.assertAlmostEqual(tone[0], 0.0, places=6)

    def test_different_frequencies(self):
        """Test that different frequencies produce different waveforms."""
        sr = 22050
        length = 1000

        tone_440 = compute_tone(440, sr, length)  # A4
        tone_880 = compute_tone(880, sr, length)  # A5 (one octave higher)

        # The waveforms should be different
        self.assertFalse(np.allclose(tone_440, tone_880))

        # A5 should complete twice as many cycles as A4 in the same time period
        # We can check this by comparing zero crossings or by correlation,
        # but for simplicity, we'll just ensure they're different

    def test_zero_frequency(self):
        """Test that a frequency of 0 produces a constant signal."""
        frequency = 0
        sr = 22050
        length = 1000

        tone = compute_tone(frequency, sr, length)

        # With frequency=0 and phi=-pi/2, all values should be 0
        expected = np.zeros(length)
        np.testing.assert_allclose(tone, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
