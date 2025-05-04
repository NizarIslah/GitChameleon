# library: torch
# version: 1.9.0
# extra_dependencies: ['scipy==1.7.3', 'numpy==1.21.6']
import torch


def erfc(input_tensor: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from scipy.special import erfc as scipy_erfc

    output = torch.from_numpy(scipy_erfc(input_tensor.numpy()))
    return output
