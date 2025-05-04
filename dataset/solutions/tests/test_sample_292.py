# Test file for sample_292.py
import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest
from unittest.mock import MagicMock, patch

import librosa
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_292 import compute_plp


class TestComputePLP(unittest.TestCase):
    """Test cases for the compute_plp function in sample_292.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a simple audio signal for testing
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        self.y = np.sin(
            2
            * np.pi
            * 440
            * np.linspace(0, self.duration, int(self.sr * self.duration))
        )

        # Common parameters
        self.hop_length = 512
        self.win_length = 1024
        self.tempo_min = 60.0
        self.tempo_max = 180.0

        # Generate onset envelope
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )

    def test_compute_plp_returns_ndarray(self):
        """Test that compute_plp returns a numpy ndarray"""
        result = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=self.tempo_min,
            tempo_max=self.tempo_max,
            onset_env=self.onset_env,
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(len(result) > 0)

    def test_compute_plp_with_none_tempo_values(self):
        """Test compute_plp with None values for tempo_min and tempo_max"""
        result = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=None,
            tempo_max=None,
            onset_env=self.onset_env,
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(len(result) > 0)

    @patch("librosa.beat.plp")
    def test_compute_plp_calls_librosa_plp(self, mock_plp):
        """Test that compute_plp correctly calls librosa.beat.plp with the right parameters"""
        # Set up the mock
        mock_plp.return_value = np.array([1.0, 2.0, 3.0])

        # Call the function
        result = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=self.tempo_min,
            tempo_max=self.tempo_max,
            onset_env=self.onset_env,
        )

        # Check that librosa.beat.plp was called with the correct parameters
        mock_plp.assert_called_once_with(
            onset_envelope=self.onset_env,
            sr=self.sr,
            tempo_min=self.tempo_min,
            tempo_max=self.tempo_max,
        )

        # Check that the result is what we expect
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_compute_plp_with_different_tempo_ranges(self):
        """Test compute_plp with different tempo ranges"""
        # Test with a narrow tempo range
        narrow_result = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=100.0,
            tempo_max=120.0,
            onset_env=self.onset_env,
        )

        # Test with a wide tempo range
        wide_result = compute_plp(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=self.win_length,
            tempo_min=30.0,
            tempo_max=300.0,
            onset_env=self.onset_env,
        )

        # Both should return valid ndarrays
        self.assertIsInstance(narrow_result, np.ndarray)
        self.assertIsInstance(wide_result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
