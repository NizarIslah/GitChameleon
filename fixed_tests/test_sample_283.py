import io
import os
import sys
import tempfile
import unittest

import librosa
import numpy as np
import soundfile as sf

# Ensure we can import sample_283 from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_283 import compute_stream


class TestSample283(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory and audio file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        
        # Generate a simple sine wave and save it to a WAV file
        self.sr = 22050  # Sample rate
        duration = 1.0   # Duration in seconds
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        sf.write(self.temp_audio_path, y, self.sr)
        
        # Parameters for testing
        self.n_fft = 1024
        self.hop_length = 512

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()
    
    def test_compute_stream(self):
        """
        Basic test for compute_stream using the real audio file instead of mocking.
        """
        # Now call compute_stream with the actual file path
        # and verify we can iterate over blocks and get STFT results.
        stream, stream_blocks = compute_stream(
            self.temp_audio_path, 
            self.sr, 
            self.n_fft, 
            self.hop_length
        )

        # Check that something is returned
        self.assertIsNotNone(stream, "Expected a non-None stream.")
        self.assertTrue(hasattr(stream, '__iter__'), "Expected 'stream' to be iterable.")

        # Check we get at least one STFT block
        self.assertIsNotNone(stream_blocks, "Expected a non-None list of STFT blocks.")
        self.assertGreater(len(stream_blocks), 0, "Expected at least one STFT block.")

        # Check the shape of the first STFT block
        # For a mono signal with n_fft=1024, the STFT should have (1 + n_fft/2) frequency bins
        first_block = stream_blocks[0]
        self.assertEqual(first_block.shape[0], 1 + self.n_fft // 2)

    def test_integration(self):
        """
        An integration-style test that checks compute_stream on real audio input
        without monkey patching.
        """
        stream, stream_blocks = compute_stream(
            self.temp_audio_path, 
            self.sr, 
            self.n_fft, 
            self.hop_length
        )

        # Basic checks
        self.assertIsNotNone(stream, "Expected a non-None stream in integration test.")
        self.assertGreater(len(stream_blocks), 0, "Expected non-empty STFT blocks.")

        # Same shape check as above
        first_block = stream_blocks[0]
        self.assertEqual(first_block.shape[0], 1 + self.n_fft // 2)

if __name__ == '__main__':
    unittest.main()