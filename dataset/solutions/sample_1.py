# library: torch
# version: 1.9.0
# extra_dependencies: ['scipy==1.7.3', 'numpy==1.21.6']
import torch


def gamma_ln(input_tensor: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from scipy.special import gammaln as scipy_gammaln

    output = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))
    return output
