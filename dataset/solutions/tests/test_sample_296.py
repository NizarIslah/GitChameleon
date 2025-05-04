import os

# Add the directory containing sample_296.py to the Python path
import sys
import unittest

import librosa
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
)

# Import the function to test
from sample_296 import compute_samples_like


class TestComputeSamplesLike(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        # Generate a simple audio signal (sine wave)
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        self.y = np.sin(
            2
            * np.pi
            * 440
            * np.linspace(0, self.duration, int(self.sr * self.duration))
        )

        # Create a spectrogram using librosa's STFT
        self.hop_length = 512
        self.n_fft = 2048
        self.D = librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length)

    def test_compute_samples_like(self):
        """Test that compute_samples_like returns the expected samples vector."""
        # Call the function under test
        result = compute_samples_like(self.y, self.sr, self.D, self.hop_length)

        # Call the librosa function directly for comparison
        expected = librosa.samples_like(self.D)

        # Check that the results are the same
        np.testing.assert_array_equal(result, expected)

        # Additional check: verify the shape of the result
        # The length of the samples vector should match the number of frames in the spectrogram
        self.assertEqual(len(result), self.D.shape[1])

        # Verify that the samples are evenly spaced by hop_length
        self.assertTrue(np.all(np.diff(result) == self.hop_length))

    def test_empty_spectrogram(self):
        """Test with an empty spectrogram."""
        # Create an empty spectrogram
        empty_D = np.array([]).reshape(0, 0)

        # Call the function under test
        result = compute_samples_like(self.y, self.sr, empty_D, self.hop_length)

        # Call the librosa function directly for comparison
        expected = librosa.samples_like(empty_D)

        # Check that the results are the same
        np.testing.assert_array_equal(result, expected)

        # The result should be an empty array
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
