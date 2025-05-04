# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_308 import compute_pyin


class TestComputePyin(unittest.TestCase):
    @patch("librosa.pyin")
    def test_compute_pyin_calls_librosa_pyin_with_correct_params(self, mock_pyin):
        # Setup mock return value
        expected_result = np.array([440.0, 442.0, 445.0])
        mock_pyin.return_value = (expected_result, None)  # librosa.pyin returns a tuple

        # Test parameters
        y = np.sin(
            2 * np.pi * 440 * np.arange(0, 1, 1 / 22050)
        )  # 1 second of 440Hz sine wave
        sr = 22050
        freq = 440
        fmin = 100
        fmax = 1000
        frame_length = 2048
        center = True
        pad_mode = "reflect"
        win_length = None
        hop_length = None
        n_thresholds = 100
        beta_parameters = (2, 18)
        boltzmann_parameter = 2
        resolution = 0.1
        max_transition_rate = 35.92
        switch_prob = 0.01
        no_trough_prob = 0.01
        fill_na = None

        # Call the function
        result = compute_pyin(
            freq=freq,
            sr=sr,
            y=y,
            fmin=fmin,
            fmax=fmax,
            frame_length=frame_length,
            center=center,
            pad_mode=pad_mode,
            win_length=win_length,
            hop_length=hop_length,
            n_thresholds=n_thresholds,
            beta_parameters=beta_parameters,
            boltzmann_parameter=boltzmann_parameter,
            resolution=resolution,
            max_transition_rate=max_transition_rate,
            switch_prob=switch_prob,
            no_trough_prob=no_trough_prob,
            fill_na=fill_na,
        )

        # Assert that librosa.pyin was called with the correct parameters
        mock_pyin.assert_called_once_with(y, fmin=fmin, fmax=fmax, center=center)

        # Assert that the result is what we expect
        np.testing.assert_array_equal(result, expected_result)

    def test_compute_pyin_with_real_audio_data(self):
        # Create a simple sine wave as test audio data
        sr = 22050
        duration = 0.5  # seconds
        freq = 440  # Hz (A4 note)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * freq * t)

        # Parameters for compute_pyin
        fmin = 100
        fmax = 1000
        frame_length = 2048
        center = True
        pad_mode = "reflect"
        win_length = None
        hop_length = None
        n_thresholds = 100
        beta_parameters = (2, 18)
        boltzmann_parameter = 2
        resolution = 0.1
        max_transition_rate = 35.92
        switch_prob = 0.01
        no_trough_prob = 0.01
        fill_na = None

        # Call the function
        result = compute_pyin(
            freq=freq,
            sr=sr,
            y=y,
            fmin=fmin,
            fmax=fmax,
            frame_length=frame_length,
            center=center,
            pad_mode=pad_mode,
            win_length=win_length,
            hop_length=hop_length,
            n_thresholds=n_thresholds,
            beta_parameters=beta_parameters,
            boltzmann_parameter=boltzmann_parameter,
            resolution=resolution,
            max_transition_rate=max_transition_rate,
            switch_prob=switch_prob,
            no_trough_prob=no_trough_prob,
            fill_na=fill_na,
        )

        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check that the estimated frequencies are close to the expected frequency (440 Hz)
        # Note: We use a tolerance because pitch estimation is not always exact
        if len(result) > 0 and not np.isnan(result).all():
            # Calculate the mean of non-NaN values
            mean_freq = np.nanmean(result)
            # Check if the mean frequency is within 10% of the expected frequency
            self.assertGreaterEqual(mean_freq, freq * 0.9)
            self.assertLessEqual(mean_freq, freq * 1.1)


if __name__ == "__main__":
    unittest.main()
