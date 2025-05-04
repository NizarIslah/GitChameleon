import os

# Add the directory containing sample_282.py to the Python path
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
)

from sample_282 import compute_extraction


class TestComputeExtraction(unittest.TestCase):
    def test_compute_extraction_returns_correct_types(self):
        # Create a simple sine wave as test audio data
        sr = 22050  # Sample rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Call the function
        mel_spec, is_float32 = compute_extraction(y, sr)

        # Check that the returned mel spectrogram is a numpy array
        self.assertIsInstance(mel_spec, np.ndarray)

        # Check that the shape is as expected (mel bands x time frames)
        self.assertEqual(len(mel_spec.shape), 2)

        # Check that the second return value is a boolean
        self.assertIsInstance(is_float32, bool)

        # In librosa 0.7.0, the default dtype for melspectrogram should be float32
        self.assertTrue(is_float32)

    def test_compute_extraction_with_different_sr(self):
        # Test with a different sample rate
        sr = 44100  # Higher sample rate
        duration = 0.5  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 880 * t)  # 880 Hz sine wave

        # Call the function
        mel_spec, is_float32 = compute_extraction(y, sr)

        # Basic checks
        self.assertIsInstance(mel_spec, np.ndarray)
        self.assertTrue(is_float32)

        # The mel spectrogram should have a different shape due to different sample rate
        # but we can't predict the exact shape without knowing all parameters
        self.assertEqual(len(mel_spec.shape), 2)

    def test_compute_extraction_with_silence(self):
        # Test with silence (all zeros)
        sr = 22050
        duration = 1.0
        y = np.zeros(int(sr * duration))

        # Call the function
        mel_spec, is_float32 = compute_extraction(y, sr)

        # Check types
        self.assertIsInstance(mel_spec, np.ndarray)
        self.assertIsInstance(is_float32, bool)

        # For silence, all values in the mel spectrogram should be very small
        # (not exactly zero due to floating point precision and log scaling)
        self.assertTrue(np.all(mel_spec <= 1e-10))


if __name__ == "__main__":
    unittest.main()
