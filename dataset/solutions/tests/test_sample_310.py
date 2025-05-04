# Add the parent directory to import sys
import os
import sys
import unittest

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_310 import compute_vqt


class TestSample310(unittest.TestCase):
    def setUp(self):
        # Create a simple sine wave as test audio data
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    def test_compute_vqt(self):
        # Test that compute_vqt returns the expected output
        vqt_output = compute_vqt(self.y, self.sr)

        # Check that the output is a numpy array
        self.assertIsInstance(vqt_output, np.ndarray)

        # Check that the output has the expected shape
        # VQT should have time frames as the second dimension
        self.assertEqual(len(vqt_output.shape), 2)

        # Compare with direct librosa.vqt call
        expected_output = librosa.vqt(self.y, sr=self.sr)
        np.testing.assert_array_equal(vqt_output, expected_output)

    def test_compute_vqt_with_empty_input(self):
        # Test with empty input
        empty_y = np.array([])

        # This should raise an error since VQT can't be computed on empty arrays
        with self.assertRaises(Exception):
            compute_vqt(empty_y, self.sr)

    def test_compute_vqt_with_different_sr(self):
        # Test with a different sample rate
        different_sr = 44100
        vqt_output = compute_vqt(self.y, different_sr)

        # Check that the output is a numpy array
        self.assertIsInstance(vqt_output, np.ndarray)

        # Compare with direct librosa.vqt call
        expected_output = librosa.vqt(self.y, sr=different_sr)
        np.testing.assert_array_equal(vqt_output, expected_output)


if __name__ == "__main__":
    unittest.main()
