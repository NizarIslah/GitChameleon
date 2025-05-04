import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_291 import compute_plp


class TestComputePLP(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic signal for testing
        self.sr = 22050  # Sample rate
        self.duration = 3  # Duration in seconds
        self.hop_length = 512  # Hop length for onset detection
        self.win_length = 2048  # Window length for PLP computation

        # Generate a simple sine wave with some noise
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        # Create a signal with beats at 120 BPM (2 Hz)
        beat_freq = 2.0  # 120 BPM = 2 Hz
        self.y = np.sin(2 * np.pi * beat_freq * t)
        # Add some noise
        self.y += 0.1 * np.random.randn(len(self.y))

        # Compute onset envelope
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )

    def test_compute_plp_returns_correct_shape(self):
        """Test that compute_plp returns an array of the correct shape."""
        tempo_min = 60
        tempo_max = 180

        plp = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=tempo_min,
            tempo_max=tempo_max,
            onset_env=self.onset_env,
        )

        # PLP should have the same length as the onset envelope
        self.assertEqual(len(plp), len(self.onset_env))

    def test_compute_plp_is_normalized(self):
        """Test that the output of compute_plp is normalized."""
        tempo_min = 60
        tempo_max = 180

        plp = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=tempo_min,
            tempo_max=tempo_max,
            onset_env=self.onset_env,
        )

        # Check that the maximum value is 1.0 (normalized)
        self.assertAlmostEqual(np.max(plp), 1.0, places=5)

        # Check that all values are non-negative
        self.assertTrue(np.all(plp >= 0))

    def test_compute_plp_with_different_parameters(self):
        """Test compute_plp with different tempo ranges."""
        # Test with narrow tempo range
        narrow_plp = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=110,
            tempo_max=130,  # Narrow range around 120 BPM
            onset_env=self.onset_env,
        )

        # Test with wide tempo range
        wide_plp = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=30,
            tempo_max=300,  # Wide range
            onset_env=self.onset_env,
        )

        # Both should return valid results
        self.assertEqual(len(narrow_plp), len(self.onset_env))
        self.assertEqual(len(wide_plp), len(self.onset_env))

        # Both should be normalized
        self.assertAlmostEqual(np.max(narrow_plp), 1.0, places=5)
        self.assertAlmostEqual(np.max(wide_plp), 1.0, places=5)

    def test_compute_plp_with_none_tempo_params(self):
        """Test compute_plp with None for tempo parameters."""
        plp = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=None,
            tempo_max=None,
            onset_env=self.onset_env,
        )

        # Should still return valid results
        self.assertEqual(len(plp), len(self.onset_env))
        self.assertAlmostEqual(np.max(plp), 1.0, places=5)
        self.assertTrue(np.all(plp >= 0))


if __name__ == "__main__":
    unittest.main()
