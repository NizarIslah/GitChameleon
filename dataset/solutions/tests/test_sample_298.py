# Test file for sample_298.py
import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_298 import compute_tone


class TestComputeTone(unittest.TestCase):
    """Test cases for the compute_tone function."""

    def test_output_type(self):
        """Test that the function returns a numpy array."""
        frequency = 440  # A4 note
        sr = 22050  # Standard sample rate
        length = 1000  # 1000 samples

        result = compute_tone(frequency, sr, length)

        self.assertIsInstance(
            result, np.ndarray, "Function should return a numpy array"
        )

    def test_output_length(self):
        """Test that the output array has the correct length."""
        frequency = 440
        sr = 22050
        length = 2000

        result = compute_tone(frequency, sr, length)

        self.assertEqual(
            len(result),
            length,
            f"Output array length should be {length}, got {len(result)}",
        )

    def test_different_frequencies(self):
        """Test that different frequencies produce different waveforms."""
        sr = 22050
        length = 1000

        # Generate tones at two different frequencies
        tone_440 = compute_tone(440, sr, length)  # A4
        tone_880 = compute_tone(880, sr, length)  # A5 (one octave higher)

        # The waveforms should be different
        self.assertFalse(
            np.array_equal(tone_440, tone_880),
            "Different frequencies should produce different waveforms",
        )

    def test_zero_frequency(self):
        """Test that a frequency of 0 produces a flat line (DC signal)."""
        sr = 22050
        length = 1000

        result = compute_tone(0, sr, length)

        # A zero frequency should produce a constant value (DC signal)
        self.assertTrue(
            np.allclose(result, result[0]),
            "Zero frequency should produce a constant value",
        )

    def test_nyquist_frequency(self):
        """Test behavior near the Nyquist frequency."""
        sr = 22050
        length = 1000
        nyquist = sr // 2

        # Generate a tone at just below the Nyquist frequency
        result = compute_tone(nyquist - 100, sr, length)

        # The result should not be all zeros
        self.assertFalse(
            np.allclose(result, 0),
            "Tone below Nyquist frequency should produce a non-zero signal",
        )


if __name__ == "__main__":
    unittest.main()
