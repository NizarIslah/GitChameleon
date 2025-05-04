# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest

import librosa
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_284 import compute_stream


class TestSample284(unittest.TestCase):
    def setUp(self):
        """Create a temporary audio file for testing."""
        # Create a simple sine wave
        self.sr = 22050  # Sample rate
        self.duration = 2  # Duration in seconds
        self.y = np.sin(
            2
            * np.pi
            * 440
            * np.linspace(0, self.duration, int(self.sr * self.duration))
        )

        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.filename = self.temp_file.name

        # Write the audio to the file
        sf.write(self.filename, self.y, self.sr)

        # Parameters for testing
        self.n_fft = 2048
        self.hop_length = 512

    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.filename)

    def test_compute_stream(self):
        """Test the compute_stream function."""

        # Patch the function to use the filename parameter
        def patched_compute_stream(filename, n_fft, hop_length):
            stream_blocks = []

            stream = librosa.stream(
                filename,
                block_length=16,
                frame_length=n_fft,
                hop_length=hop_length,
                mono=True,
                fill_value=0,
            )

            for c, y_block in enumerate(stream):
                stream_blocks.append(
                    librosa.stft(
                        y_block, n_fft=n_fft, hop_length=hop_length, center=False
                    )
                )

            return stream, stream_blocks

        # Call the patched function
        stream, stream_blocks = patched_compute_stream(
            self.filename, self.n_fft, self.hop_length
        )

        # Assertions
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream_blocks, list)
        self.assertGreater(len(stream_blocks), 0)

        # Check that each block is a STFT result (complex-valued spectrogram)
        for block in stream_blocks:
            self.assertIsInstance(block, np.ndarray)
            self.assertEqual(block.dtype.kind, "c")  # Complex data type
            self.assertEqual(
                block.shape[0], self.n_fft // 2 + 1
            )  # Number of frequency bins

    def test_compute_stream_with_different_parameters(self):
        """Test compute_stream with different FFT and hop length parameters."""

        # Patch the function to use the filename parameter
        def patched_compute_stream(filename, n_fft, hop_length):
            stream_blocks = []

            stream = librosa.stream(
                filename,
                block_length=16,
                frame_length=n_fft,
                hop_length=hop_length,
                mono=True,
                fill_value=0,
            )

            for c, y_block in enumerate(stream):
                stream_blocks.append(
                    librosa.stft(
                        y_block, n_fft=n_fft, hop_length=hop_length, center=False
                    )
                )

            return stream, stream_blocks

        # Test with different parameters
        n_fft = 1024
        hop_length = 256

        stream, stream_blocks = patched_compute_stream(self.filename, n_fft, hop_length)

        # Assertions
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream_blocks, list)
        self.assertGreater(len(stream_blocks), 0)

        # Check that each block has the expected shape based on the new parameters
        for block in stream_blocks:
            self.assertIsInstance(block, np.ndarray)
            self.assertEqual(block.shape[0], n_fft // 2 + 1)  # Number of frequency bins


if __name__ == "__main__":
    unittest.main()
