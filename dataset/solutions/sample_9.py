# library: torch
# version: 1.10.0
# extra_dependencies: []
import torch


def bessel_i0(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.special.i0(input_tensor)
