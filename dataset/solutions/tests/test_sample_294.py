import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_294 import compute_times_like


class TestComputeTimesLike(unittest.TestCase):
    def setUp(self):
        # Create a simple audio signal for testing
        self.sr = 22050  # Sample rate in Hz
        self.duration = 1.0  # Duration in seconds
        self.hop_length = 512  # Hop length for STFT

        # Generate a simple sine wave as test audio
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Compute a spectrogram for testing
        self.D = np.abs(librosa.stft(self.y, hop_length=self.hop_length))

    def test_compute_times_like(self):
        """Test that compute_times_like returns the correct times vector."""
        # Call the function under test
        times = compute_times_like(self.y, self.sr, self.hop_length, self.D)

        # Expected result: librosa.times_like should return frame times in seconds
        expected_times = librosa.times_like(self.D, sr=self.sr)

        # Check that the output is a numpy array
        self.assertIsInstance(times, np.ndarray)

        # Check that the shape matches the expected shape
        self.assertEqual(times.shape, expected_times.shape)

        # Check that the values match the expected values
        np.testing.assert_allclose(times, expected_times)

        # Check that the length of times matches the number of frames in D
        self.assertEqual(len(times), self.D.shape[1])

        # Check that the times are evenly spaced
        if len(times) > 1:
            time_diffs = np.diff(times)
            self.assertTrue(np.allclose(time_diffs, time_diffs[0]))

            # Check that the time step is approximately hop_length/sr
            expected_step = self.hop_length / self.sr
            self.assertAlmostEqual(time_diffs[0], expected_step, places=5)

    def test_with_different_sr(self):
        """Test compute_times_like with a different sample rate."""
        # Use a different sample rate
        new_sr = 44100

        # Call the function under test
        times = compute_times_like(self.y, new_sr, self.hop_length, self.D)

        # Expected result with the new sample rate
        expected_times = librosa.times_like(self.D, sr=new_sr)

        # Check that the values match the expected values
        np.testing.assert_allclose(times, expected_times)

        # The time step should be proportional to 1/sr
        if len(times) > 1:
            time_step = times[1] - times[0]
            expected_step = self.hop_length / new_sr
            self.assertAlmostEqual(time_step, expected_step, places=5)


if __name__ == "__main__":
    unittest.main()
