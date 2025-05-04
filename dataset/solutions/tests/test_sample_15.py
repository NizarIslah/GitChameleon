# Add the parent directory to import sys
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_15 import stft


class TestSTFT(unittest.TestCase):
    """Test cases for the stft function in sample_15.py."""

    def test_basic_functionality(self):
        """Test basic functionality of the stft function."""
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1  # seconds
        frequency = 440  # Hz (A4 note)
        t = torch.arange(0, duration, 1.0 / sample_rate)
        audio_signal = torch.sin(2 * torch.pi * frequency * t)

        n_fft = 512
        result = stft(audio_signal, n_fft)

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check the shape of the output
        # For real view of complex output, the shape should be (n_fft//2 + 1, num_frames, 2)
        # where num_frames depends on the length of the input and the hop_length (default is n_fft//4)
        expected_freq_bins = n_fft // 2 + 1
        self.assertEqual(result.shape[0], expected_freq_bins)
        self.assertEqual(result.shape[2], 2)  # Real and imaginary parts

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, audio_signal.dtype)

    def test_different_n_fft_values(self):
        """Test stft with different n_fft values."""
        # Create a simple audio signal
        audio_signal = torch.randn(16000)

        # Test with different n_fft values
        n_fft_values = [256, 512, 1024]

        for n_fft in n_fft_values:
            result = stft(audio_signal, n_fft)

            # Check the shape
            expected_freq_bins = n_fft // 2 + 1
            self.assertEqual(result.shape[0], expected_freq_bins)
            self.assertEqual(result.shape[2], 2)  # Real and imaginary parts

    def test_multi_channel_audio(self):
        """Test stft with multi-channel audio."""
        # Create a stereo audio signal (2 channels)
        num_samples = 16000
        num_channels = 2
        audio_signal = torch.randn(num_channels, num_samples)

        n_fft = 512

        # PyTorch's stft expects the input to be (..., time), so we need to handle each channel separately
        # Our wrapper doesn't handle this automatically, so we'll test by applying to each channel

        # Process first channel
        result_ch1 = stft(audio_signal[0], n_fft)

        # Process second channel
        result_ch2 = stft(audio_signal[1], n_fft)

        # Check shapes
        expected_freq_bins = n_fft // 2 + 1
        self.assertEqual(result_ch1.shape[0], expected_freq_bins)
        self.assertEqual(result_ch2.shape[0], expected_freq_bins)

        # Check that the two channels produce different results (they should, as they're random)
        self.assertFalse(torch.allclose(result_ch1, result_ch2))

    def test_empty_audio_signal(self):
        """Test stft with an empty audio signal."""
        audio_signal = torch.tensor([])
        n_fft = 512

        # This should raise an error since STFT can't be computed on an empty signal
        with self.assertRaises(RuntimeError):
            stft(audio_signal, n_fft)

    def test_compare_with_torch_stft(self):
        """Test that our stft function matches torch.stft with view_as_real."""
        audio_signal = torch.randn(16000)
        n_fft = 512

        # Our implementation
        result = stft(audio_signal, n_fft)

        # Direct torch.stft call with the same parameters
        expected = torch.view_as_real(
            torch.stft(audio_signal, n_fft=n_fft, return_complex=True)
        )

        # Check that they match exactly
        torch.testing.assert_close(result, expected)

        # Check shapes match
        self.assertEqual(result.shape, expected.shape)

    def test_different_dtypes(self):
        """Test stft with different dtypes."""
        # Test with float32
        audio_signal_f32 = torch.randn(16000, dtype=torch.float32)
        n_fft = 512
        result_f32 = stft(audio_signal_f32, n_fft)
        self.assertEqual(result_f32.dtype, torch.float32)

        # Test with float64
        audio_signal_f64 = torch.randn(16000, dtype=torch.float64)
        result_f64 = stft(audio_signal_f64, n_fft)
        self.assertEqual(result_f64.dtype, torch.float64)

    def test_non_tensor_input(self):
        """Test stft with non-tensor input (should raise AttributeError)."""
        with self.assertRaises(AttributeError):
            # List is not a tensor and has no 'dim' attribute
            stft([1, 2, 3, 4], 512)

    def test_non_integer_n_fft(self):
        """Test stft with non-integer n_fft (should raise TypeError)."""
        audio_signal = torch.randn(16000)

        with self.assertRaises(TypeError):
            # n_fft must be an integer
            stft(audio_signal, 512.5)

    def test_negative_n_fft(self):
        """Test stft with negative n_fft (should raise RuntimeError)."""
        audio_signal = torch.randn(16000)

        with self.assertRaises(RuntimeError):
            # n_fft must be positive
            stft(audio_signal, -512)

    def test_zero_n_fft(self):
        """Test stft with zero n_fft (should raise RuntimeError)."""
        audio_signal = torch.randn(16000)

        with self.assertRaises(RuntimeError):
            # n_fft must be positive
            stft(audio_signal, 0)

    def test_complex_output_structure(self):
        """Test that the output has the correct structure for a real view of complex tensor."""
        audio_signal = torch.randn(16000)
        n_fft = 512
        result = stft(audio_signal, n_fft)

        # Get the direct complex output
        complex_output = torch.stft(audio_signal, n_fft=n_fft, return_complex=True)

        # Check that the real part in our result matches the real part of the complex output
        torch.testing.assert_close(result[..., 0], complex_output.real)

        # Check that the imaginary part in our result matches the imaginary part of the complex output
        torch.testing.assert_close(result[..., 1], complex_output.imag)


if __name__ == "__main__":
    unittest.main()
