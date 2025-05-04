import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
import librosa
import sys
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_283


class TestComputeStream(unittest.TestCase):
    def setUp(self):
        # Create a temporary audio file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")

        # Generate a simple sine wave
        sr = 22050  # Sample rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save as WAV file
        sf.write(self.audio_path, y, sr)

        # Parameters for testing
        self.y = y
        self.sr = sr
        self.n_fft = 1024
        self.hop_length = 512

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_compute_stream_returns_expected_types(self):
        """Test that compute_stream returns the expected types."""
        stream, stream_blocks = sample_283.compute_stream(
            self.audio_path, self.y, self.sr, self.n_fft, self.hop_length
        )

        # Check that stream is a generator
        self.assertTrue(hasattr(stream, "__iter__"))

        # Check that stream_blocks is a list
        self.assertIsInstance(stream_blocks, list)

        # Check that stream_blocks contains STFT matrices
        for block in stream_blocks:
            self.assertIsInstance(block, np.ndarray)
            self.assertEqual(block.shape[0], self.n_fft // 2 + 1)  # STFT output size

    def test_compute_stream_processes_blocks(self):
        """Test that compute_stream processes blocks correctly."""
        # Count the number of blocks that should be processed
        with sf.SoundFile(self.audio_path) as f:
            file_duration = len(f) / f.samplerate
            blocksize = self.n_fft + 15 * self.hop_length
            overlap = self.n_fft - self.hop_length
            effective_blocksize = blocksize - overlap
            expected_blocks = max(1, int(np.ceil(len(f) / effective_blocksize)))

        # Get actual blocks
        _, stream_blocks = sample_283.compute_stream(
            self.audio_path, self.y, self.sr, self.n_fft, self.hop_length
        )

        # Check number of blocks
        self.assertEqual(len(stream_blocks), expected_blocks)


if __name__ == "__main__":
    unittest.main()
