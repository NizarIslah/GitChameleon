# library: torch
# version: 1.9.0
# extra_dependencies: ['scipy==1.7.3', 'numpy==1.21.6']
import torch


def bessel_i1(input_tensor: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from scipy.special import i1 as scipy_i1

    output = torch.from_numpy(scipy_i1(input_tensor.numpy()))
    return output
