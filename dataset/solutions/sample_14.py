# library: torch
# version: 1.13
# extra_dependencies: []
import torch


def stft(audio_signal: torch.Tensor, n_fft: int) -> torch.Tensor:
    return torch.stft(audio_signal, n_fft=n_fft, return_complex=False)
