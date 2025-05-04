# library: torch
# version: 2
# extra_dependencies: []
import torch


def stft(audio_signal: torch.Tensor, n_fft: int) -> torch.Tensor:
    return torch.view_as_real(
        torch.stft(audio_signal, n_fft=n_fft, return_complex=True)
    )
