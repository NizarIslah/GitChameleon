# Add the parent directory to import sys
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_16 import istft


class TestISTFT(unittest.TestCase):
    """Test cases for the istft function in sample_16.py."""

    def test_basic_functionality(self):
        """Test basic functionality of the istft function."""
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1  # seconds
        frequency = 440  # Hz (A4 note)
        t = torch.arange(0, duration, 1.0 / sample_rate)
        audio_signal = torch.sin(2 * torch.pi * frequency * t)

        # Parameters for STFT and ISTFT
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        # Compute STFT
        spectrogram = torch.stft(
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )

        # Apply ISTFT
        reconstructed = istft(
            spectrogram,
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        # Check that the result is a tensor
        self.assertIsInstance(reconstructed, torch.Tensor)

        # Check that the shape matches the original signal
        self.assertEqual(reconstructed.shape, audio_signal.shape)

        # Check that the dtype is preserved
        self.assertEqual(reconstructed.dtype, audio_signal.dtype)

        # Check that the reconstructed signal is close to the original
        # Note: There will be some differences due to the windowing and overlap-add process
        self.assertTrue(
            torch.allclose(reconstructed, audio_signal, rtol=1e-2, atol=1e-2)
        )

    def test_different_parameter_values(self):
        """Test istft with different parameter values."""
        # Create a simple audio signal
        audio_signal = torch.randn(16000)

        # Test with different parameter combinations
        parameter_sets = [
            {"n_fft": 256, "hop_length": 64, "win_length": 256},
            {"n_fft": 512, "hop_length": 128, "win_length": 512},
            {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
        ]

        for params in parameter_sets:
            n_fft = params["n_fft"]
            hop_length = params["hop_length"]
            win_length = params["win_length"]

            # Compute STFT
            spectrogram = torch.stft(
                audio_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length),
                return_complex=True,
            )

            # Apply ISTFT
            reconstructed = istft(
                spectrogram,
                audio_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

            # Check shape
            self.assertEqual(reconstructed.shape, audio_signal.shape)

            # Check that the reconstructed signal is reasonably close to the original
            self.assertTrue(
                torch.allclose(reconstructed, audio_signal, rtol=1e-2, atol=1e-2)
            )

    def test_multi_channel_audio(self):
        """Test istft with multi-channel audio."""
        # Create a stereo audio signal (2 channels)
        num_samples = 16000
        num_channels = 2
        audio_signal = torch.randn(num_channels, num_samples)

        # Parameters for STFT and ISTFT
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        # Process each channel separately
        for channel in range(num_channels):
            # Compute STFT for this channel
            spectrogram = torch.stft(
                audio_signal[channel],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length),
                return_complex=True,
            )

            # Apply ISTFT
            reconstructed = istft(
                spectrogram,
                audio_signal[channel],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

            # Check shape
            self.assertEqual(reconstructed.shape, audio_signal[channel].shape)

            # Check that the reconstructed signal is reasonably close to the original
            self.assertTrue(
                torch.allclose(
                    reconstructed, audio_signal[channel], rtol=1e-2, atol=1e-2
                )
            )

    def test_compare_with_torch_istft(self):
        """Test that our istft function matches torch.istft with the same parameters."""
        audio_signal = torch.randn(16000)

        # Parameters for STFT and ISTFT
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        # Compute STFT
        spectrogram = torch.stft(
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )

        # Our implementation
        result = istft(
            spectrogram,
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        # Direct torch.istft call with the same parameters
        expected = torch.istft(
            spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            length=audio_signal.shape[0],
            normalized=False,
        )

        # Check that they match exactly
        torch.testing.assert_close(result, expected)

        # Check shapes match
        self.assertEqual(result.shape, expected.shape)

    def test_different_dtypes(self):
        """Test istft with different dtypes."""
        # Test with float32
        audio_signal_f32 = torch.randn(16000, dtype=torch.float32)

        # Parameters for STFT and ISTFT
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        # Compute STFT
        spectrogram_f32 = torch.stft(
            audio_signal_f32,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, dtype=torch.float32),
            return_complex=True,
        )

        # Apply ISTFT
        result_f32 = istft(
            spectrogram_f32,
            audio_signal_f32,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        self.assertEqual(result_f32.dtype, torch.float32)

        # Test with float64
        audio_signal_f64 = torch.randn(16000, dtype=torch.float64)

        # Compute STFT
        spectrogram_f64 = torch.stft(
            audio_signal_f64,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, dtype=torch.float64),
            return_complex=True,
        )

        # Apply ISTFT
        result_f64 = istft(
            spectrogram_f64,
            audio_signal_f64,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        self.assertEqual(result_f64.dtype, torch.float64)

    def test_non_tensor_input(self):
        """Test istft with non-tensor input (should raise TypeError)."""
        # Parameters for ISTFT
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        # Create a valid spectrogram and signal
        audio_signal = torch.randn(16000)
        spectrogram = torch.stft(
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )

        # Test with non-tensor spectrogram
        with self.assertRaises(TypeError):
            istft(
                spectrogram.cpu().numpy(),  # Convert to numpy array
                audio_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

        # Test with non-tensor signal
        with self.assertRaises(AttributeError):
            istft(
                spectrogram,
                audio_signal.numpy().tolist(),  # Convert to list
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

    def test_invalid_parameter_values(self):
        """Test istft with invalid parameter values."""
        # Create a valid spectrogram and signal
        audio_signal = torch.randn(16000)
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        spectrogram = torch.stft(
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )

        # Test with negative n_fft
        with self.assertRaises(RuntimeError):
            istft(
                spectrogram,
                audio_signal,
                n_fft=-512,
                hop_length=hop_length,
                win_length=win_length,
            )

        # Test with zero hop_length
        with self.assertRaises(RuntimeError):
            istft(
                spectrogram,
                audio_signal,
                n_fft=n_fft,
                hop_length=0,
                win_length=win_length,
            )

        # Test with win_length > n_fft
        with self.assertRaises(RuntimeError):
            istft(
                spectrogram,
                audio_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft + 1,
            )

    def test_round_trip_transformation(self):
        """Test round-trip STFT -> ISTFT transformation."""
        # Create a simple audio signal
        audio_signal = torch.randn(16000)

        # Parameters for STFT and ISTFT
        n_fft = 512
        hop_length = n_fft // 4
        win_length = n_fft

        # Compute STFT
        spectrogram = torch.stft(
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True,
        )

        # Apply ISTFT
        reconstructed = istft(
            spectrogram,
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        # Check that the reconstructed signal is close to the original
        # Note: There will be some differences due to the windowing and overlap-add process
        self.assertTrue(
            torch.allclose(reconstructed, audio_signal, rtol=1e-2, atol=1e-2)
        )

        # Compute the signal-to-noise ratio (SNR) as a measure of reconstruction quality
        noise = audio_signal - reconstructed
        signal_power = torch.mean(audio_signal**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(signal_power / noise_power)

        # Check that the SNR is reasonably high (typically > 20 dB for good reconstruction)
        self.assertGreater(snr, 20.0)


if __name__ == "__main__":
    unittest.main()
