# library: torch
# version: 1.10.0
# extra_dependencies: []
import torch


def gamma_ln(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.special.gammaln(input_tensor)
