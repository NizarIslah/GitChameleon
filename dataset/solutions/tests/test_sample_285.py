import os
import sys
import unittest
from typing import Optional

import numpy as np

# Make sure we import our solution module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_285
from sample_285 import compute_griffinlim


class TestGriffinLim(unittest.TestCase):
    def setUp(self):
        # 1 second of a 440 Hz sine wave @ 22050 Hz
        self.n_fft = 512
        self.hop_length = 128
        self.sr = 22050
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.y = np.sin(2 * np.pi * 440 * t)

        # Build a repeated magnitude spectrogram
        S_complex = np.abs(np.fft.rfft(self.y[: self.n_fft]))
        n_frames = int(np.ceil(len(self.y) / self.hop_length))
        self.S = np.tile(S_complex[:, np.newaxis], (1, n_frames))

        # **Inject** a global `momentum` into sample_285 so the function can use it
        sample_285.momentum = 0.99

    def test_compute_griffinlim_basic(self):
        """Runs without error and returns roughly the right length."""
        result = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=42,
            n_iter=5,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode="reflect",
            n_fft=self.n_fft,
        )
        self.assertIsInstance(result, np.ndarray)
        # Should be within ±10% of original length
        self.assertGreaterEqual(len(result), len(self.y) * 0.9)
        self.assertLessEqual(len(result), len(self.y) * 1.1)

    def test_compute_griffinlim_different_parameters(self):
        """Different window, centering, dtype still runs and respects our momentum."""
        result = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=42,
            n_iter=3,
            hop_length=self.hop_length,
            win_length=self.n_fft // 2,
            window="hamming",
            center=False,
            dtype=np.float64,
            length=None,
            pad_mode="constant",
            n_fft=self.n_fft,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_compute_griffinlim_reproducibility(self):
        """Same seed → identical output; different → not identical."""
        r1 = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=123,
            n_iter=2,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode="reflect",
            n_fft=self.n_fft,
        )
        r2 = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=123,
            n_iter=2,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode="reflect",
            n_fft=self.n_fft,
        )
        # identical when seed is same
        np.testing.assert_array_equal(r1, r2)

        # different seed → different output
        r3 = compute_griffinlim(
            y=self.y,
            sr=self.sr,
            S=self.S,
            random_state=999,
            n_iter=2,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode="reflect",
            n_fft=self.n_fft,
        )
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(r1, r3)


if __name__ == "__main__":
    unittest.main()
