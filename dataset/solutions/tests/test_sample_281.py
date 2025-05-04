import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_281 import compute_extraction


class TestSample281(unittest.TestCase):
    def test_compute_extraction_output_types(self):
        """Test that compute_extraction returns the expected output types."""
        # Create a simple audio signal (sine wave)
        sr = 22050  # Sample rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Call the function
        mel_spec, is_float64 = compute_extraction(y, sr)

        # Check that the first return value is a numpy ndarray
        self.assertIsInstance(mel_spec, np.ndarray)

        # Check that the second return value is a boolean
        self.assertIsInstance(is_float64, bool)

    def test_compute_extraction_dtype_check(self):
        """Test that the dtype check in compute_extraction works correctly."""
        # Create a simple audio signal
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)

        # Call the function
        mel_spec, is_float64 = compute_extraction(y, sr)

        # Verify that the second return value correctly identifies the dtype
        self.assertEqual(is_float64, (mel_spec.dtype == np.float64))

        # Since librosa.feature.melspectrogram should return float64 by default,
        # is_float64 should be True
        self.assertTrue(is_float64)

    def test_compute_extraction_shape(self):
        """Test that the melspectrogram has the expected shape."""
        # Create a simple audio signal
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)

        # Call the function
        mel_spec, _ = compute_extraction(y, sr)

        # Check that the melspectrogram has the expected shape
        # The first dimension should be the number of mel bands (default is 128)
        self.assertEqual(mel_spec.shape[0], 128)

        # The second dimension should depend on the length of the input signal
        # and the hop length (default is 512), but it should be greater than 0
        self.assertGreater(mel_spec.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
