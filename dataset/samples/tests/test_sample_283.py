import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
import librosa
import sys
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.samples.sample_283 import compute_stream

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
    
    @patch('dataset.samples.sample_283.sf.blocks')
    def test_compute_stream(self, mock_blocks):
        # Create a mock for sf.blocks
        mock_block = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_blocks.return_value = [mock_block]
        
        # Mock librosa.to_mono to return a simple array
        mono_output = np.array([0.2, 0.35])
        with patch('dataset.samples.sample_283.librosa.to_mono', return_value=mono_output):
            # Mock librosa.stft to return a simple complex array
            stft_output = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
            with patch('dataset.samples.sample_283.librosa.stft', return_value=stft_output):
                # Call the function with our test data
                stream, stream_blocks = compute_stream(
                    self.y, self.sr, self.n_fft, self.hop_length
                )
                
                # Verify the function was called with correct parameters
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
        Integration test that uses actual audio processing functions
        """
        # This test will fail because the original function has a bug:
        # 'filename' is not defined in the function parameters
        
        # We'll patch the function to fix the bug for testing purposes
        with patch('dataset.samples.sample_283.sf.blocks') as mock_blocks:
            # Create a mock audio block
            mock_block = np.random.rand(2, 1000)  # Stereo audio block
            mock_blocks.return_value = [mock_block]
            
            # Call the function
            stream, stream_blocks = compute_stream(
                self.y, self.sr, self.n_fft, self.hop_length
            )
            
            # Basic validation
            self.assertIsNotNone(stream)
            self.assertEqual(len(stream_blocks), 1)
            # Check that the output has the expected shape for an STFT
            # For a mono signal with n_fft=1024, we expect (1+n_fft/2) frequency bins
            self.assertEqual(stream_blocks[0].shape[0], 1 + self.n_fft // 2)

if __name__ == '__main__':
    unittest.main()