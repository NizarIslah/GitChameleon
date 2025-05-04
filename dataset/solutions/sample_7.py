# library: torch
# version: 1.10.0
# extra_dependencies: []
import torch


def erf(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.special.erf(input_tensor)
