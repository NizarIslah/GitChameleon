import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import librosa
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_283 import compute_stream


class TestSample283(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary audio file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        
        # Generate a simple sine wave
        sr = 22050  # Sample rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV file
        sf.write(self.temp_audio_path, y, sr)
        
        # Parameters for testing
        self.y = y
        self.sr = sr
        self.n_fft = 1024
        self.hop_length = 512
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_integration(self):
        """
        Integration test that uses actual audio processing functions
        """
        # Call the function with the filename
        stream, stream_blocks = compute_stream(
            self.temp_audio_path, self.sr, self.n_fft, self.hop_length
        )
        # Basic validation
        self.assertIsNotNone(stream)
        self.assertTrue(len(stream_blocks) > 0)
        # Check that the output has the expected shape for an STFT
        # For a mono signal with n_fft=1024, we expect (1+n_fft/2) frequency bins
        self.assertEqual(stream_blocks[0].shape[0], 1 + self.n_fft // 2)

if __name__ == '__main__':
    unittest.main()
