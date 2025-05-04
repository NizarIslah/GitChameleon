# library: torch
# version: 1.13
# extra_dependencies: []
import torch


def invert_mask(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.BoolTensor:
    return ~(tensor1 < tensor2).bool()
