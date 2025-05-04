# library: torch
# version: 1.13
# extra_dependencies: []
import torch


def istft(
    spectrogram: torch.Tensor,
    signal: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized=False,
) -> torch.Tensor:
    return torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        length=signal.shape[0],
        normalized=False,
    )
