import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    @patch('sample_283.sf.blocks')
    def test_compute_stream(self, mock_blocks):
        """
        Updated so it calls compute_stream with a file path (rather than the raw array).
        This fixes the NameError in the function that references 'filename'.
        """
        mock_block = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_blocks.return_value = [mock_block]
        
        mono_output = np.array([0.2, 0.35])
        with patch('sample_283.librosa.to_mono', return_value=mono_output):
            stft_output = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
            with patch('sample_283.librosa.stft', return_value=stft_output):
                stream, stream_blocks = compute_stream(
                    self.temp_audio_path,  # Pass the actual file path
                    self.sr, 
                    self.n_fft, 
                    self.hop_length
                )
                
                mock_blocks.assert_called_once()
                blocksize_arg = self.n_fft + 15 * self.hop_length
                overlap_arg = self.n_fft - self.hop_length
                self.assertEqual(mock_blocks.call_args[1]['blocksize'], blocksize_arg)
                self.assertEqual(mock_blocks.call_args[1]['overlap'], overlap_arg)
                self.assertEqual(mock_blocks.call_args[1]['fill_value'], 0)
                
                # Check the results
                self.assertEqual(stream, mock_blocks.return_value)
                self.assertEqual(len(stream_blocks), 1)
                np.testing.assert_array_equal(stream_blocks[0], stft_output)

    def test_integration(self):
        """
        Integration test that uses actual audio processing functions.
        Updated to pass a filename to the function under test, since
        the function references 'filename'.
        """
        stream, stream_blocks = compute_stream(
            self.temp_audio_path,  # Pass the path instead of the raw array
            self.sr,
            self.n_fft,
            self.hop_length
        )
        # Basic validation
        self.assertIsNotNone(stream)
        self.assertTrue(len(stream_blocks) >= 1)
        # For a mono STFT with n_fft=1024, each STFT block should have (1+n_fft/2) freq bins
        # but exact sizes can vary by block, so just confirm it matches that pattern:
        for block in stream_blocks:
            self.assertEqual(block.shape[0], 1 + self.n_fft // 2)


if __name__ == '__main__':
    unittest.main()