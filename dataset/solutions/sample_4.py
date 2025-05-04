# library: torch
# version: 1.9.0
# extra_dependencies: ['scipy==1.7.3', 'numpy==1.21.6']
import torch


def bessel_i0(input_tensor: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from scipy.special import i0 as scipy_i0

    output = torch.from_numpy(scipy_i0(input_tensor.numpy()))
    return output
