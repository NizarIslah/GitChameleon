# library: torch
# version: 1.10.0
# extra_dependencies: []
import torch


def erfc(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.special.erfc(input_tensor)
