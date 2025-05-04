# test_sample.py

import os
import sys
import unittest
from typing import Tuple

import librosa
import numpy as np
import scipy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_307 import compute_pyin


class TestComputePYIN(unittest.TestCase):
    """Test cases for the compute_pyin function."""

    def setUp(self):
        """Set up test data."""
        # Create a simple sine wave as test audio
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        self.freq = 440.0  # A4 note frequency

        # Generate a sine wave
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.y = 0.5 * np.sin(2 * np.pi * self.freq * t)

        # Default parameters for testing
        self.fmin = 50
        self.fmax = 2000
        self.frame_length = 2048
        self.win_length = 1024
        self.hop_length = 512
        self.center = True
        self.pad_mode = "reflect"
        self.n_thresholds = 100
        self.beta_parameters = (2, 18)
        self.boltzmann_parameter = 2
        self.resolution = 0.1
        self.max_transition_rate = 35.92
        self.switch_prob = 0.01
        self.no_trough_prob = 0.01
        self.fill_na = None

    def test_basic_functionality(self):
        """Test that the function runs without errors and returns expected shape."""
        f0 = compute_pyin(
            freq=self.freq,
            sr=self.sr,
            y=self.y,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=self.fill_na,
        )

        # Check that output is a numpy array
        self.assertIsInstance(f0, np.ndarray)

        # Compute expected number of frames using standard centered framing logic:
        # n_frames = floor((len(y) + 2*pad - frame_length) / hop_length) + 1, where pad = frame_length//2 if center=True
        pad = self.frame_length // 2 if self.center else 0
        expected_frames = (
            (len(self.y) + 2 * pad - self.frame_length) // self.hop_length
        ) + 1

        self.assertEqual(len(f0), expected_frames)

    def test_frequency_estimation(self):
        """Test that the function estimates frequencies close to the input frequency."""
        f0 = compute_pyin(
            freq=self.freq,
            sr=self.sr,
            y=self.y,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=0.0,  # Use 0.0 to fill NA values for this test
        )

        # Filter out unvoiced frames (zeros)
        voiced_frames = f0[f0 > 0]

        # Check that we have some voiced frames
        self.assertGreater(len(voiced_frames), 0)

        # Check that the mean estimated frequency is close to the input frequency
        # Allow for a 10% error margin
        if len(voiced_frames) > 0:
            mean_f0 = np.mean(voiced_frames)
            self.assertLess(abs(mean_f0 - self.freq) / self.freq, 0.1)

    def test_fill_na_parameter(self):
        """Test that the fill_na parameter works as expected."""
        # Test with fill_na = None
        f0_none = compute_pyin(
            freq=self.freq,
            sr=self.sr,
            y=self.y,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=None,
        )

        # Test with fill_na = -1
        f0_neg1 = compute_pyin(
            freq=self.freq,
            sr=self.sr,
            y=self.y,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=-1,
        )

        # Check that unvoiced frames are filled with -1 when fill_na=-1
        self.assertTrue(np.any(f0_neg1 == -1))

    def test_parameter_defaults(self):
        """Test that the function handles default parameters correctly."""
        # Test with win_length and hop_length as None
        f0 = compute_pyin(
            freq=self.freq,
            sr=self.sr,
            y=self.y,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=None,  # Should default to frame_length // 2
            hop_length=None,  # Should default to frame_length // 4
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=self.fill_na,
        )

        # Check that output is a numpy array
        self.assertIsInstance(f0, np.ndarray)

        # Expected hop length with default = frame_length // 4
        expected_hop_length = self.frame_length // 4

        # Again, use the centered framing logic
        pad = self.frame_length // 2 if self.center else 0
        expected_frames = (
            (len(self.y) + 2 * pad - self.frame_length) // expected_hop_length
        ) + 1

        self.assertEqual(len(f0), expected_frames)

    def test_different_audio_inputs(self):
        """Test the function with different audio inputs."""
        # Test with a different frequency
        freq2 = 880.0  # A5 note
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        y2 = 0.5 * np.sin(2 * np.pi * freq2 * t)

        f0 = compute_pyin(
            freq=freq2,
            sr=self.sr,
            y=y2,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=0.0,
        )

        # Filter out unvoiced frames
        voiced_frames = f0[f0 > 0]

        # Check that we have some voiced frames
        self.assertGreater(len(voiced_frames), 0)

        # Check that the mean estimated frequency is close to the input frequency
        if len(voiced_frames) > 0:
            mean_f0 = np.mean(voiced_frames)
            self.assertLess(
                abs(mean_f0 - freq2) / freq2, 0.2
            )  # Allow for a 20% error margin

    def test_edge_cases(self):
        """Test edge cases for the function."""
        # Test with a very short signal
        short_y = np.sin(
            2 * np.pi * self.freq * np.linspace(0, 0.1, int(self.sr * 0.1))
        )

        # Use smaller frame_length for short signal
        short_frame_length = 512
        short_win_length = 256
        short_hop_length = 128

        f0 = compute_pyin(
            freq=self.freq,
            sr=self.sr,
            y=short_y,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=short_frame_length,
            center=self.center,
            pad_mode=self.pad_mode,
            win_length=short_win_length,
            hop_length=short_hop_length,
            n_thresholds=self.n_thresholds,
            beta_parameters=self.beta_parameters,
            boltzmann_parameter=self.boltzmann_parameter,
            resolution=self.resolution,
            max_transition_rate=self.max_transition_rate,
            switch_prob=self.switch_prob,
            no_trough_prob=self.no_trough_prob,
            fill_na=0.0,
        )

        # Check that output is a numpy array
        self.assertIsInstance(f0, np.ndarray)

        # Check that we get some output even for a short signal
        self.assertGreater(len(f0), 0)


if __name__ == "__main__":
    unittest.main()
