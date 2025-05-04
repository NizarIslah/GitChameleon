# library: torch
# version: 1.12.0
# extra_dependencies: []
import torch


def log_ndtr(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.special.log_ndtr(input_tensor)
