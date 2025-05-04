import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_312 import compute_griffinlim_cqt


class TestSample312(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility
        np.random.seed(42)

        # Create a simple audio signal for testing
        self.sr = 22050  # Sample rate
        self.duration = 1.0  # Duration in seconds
        self.y = np.sin(
            2
            * np.pi
            * 440
            * np.linspace(0, self.duration, int(self.sr * self.duration))
        )

        # Generate a CQT spectrogram
        self.bins_per_octave = 12
        self.n_bins = 84
        self.fmin = librosa.note_to_hz("C2")
        self.C = librosa.cqt(
            self.y,
            sr=self.sr,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin,
        )

    def test_compute_griffinlim_cqt(self):
        """Test that compute_griffinlim_cqt returns an audio signal of expected shape."""
        # Call the function with minimal required parameters
        # Note: The function signature has many parameters but the implementation only uses a few
        result = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=30,
            hop_length=512,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=None,
            momentum=0.99,
            init=None,
        )

        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check that the result has the expected dimensions (1D audio signal)
        self.assertEqual(len(result.shape), 1)

        # The length of the reconstructed signal might not match the original exactly,
        # but it should be close to the original length
        self.assertGreater(len(result), 0)

    def test_with_random_init(self):
        """Test compute_griffinlim_cqt with a random initialization."""
        # Call the function with 'random' initialization
        result = compute_griffinlim_cqt(
            y=self.y,
            sr=self.sr,
            C=self.C,
            n_iter=30,
            hop_length=512,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="reflect",
            res_type="kaiser_best",
            dtype=np.float32,
            length=None,
            momentum=0.99,
            init="random",
        )

        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check that the result has the expected dimensions (1D audio signal)
        self.assertEqual(len(result.shape), 1)


if __name__ == "__main__":
    unittest.main()
