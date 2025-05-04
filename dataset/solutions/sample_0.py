# library: torch
# version: 1.9.0
# extra_dependencies: ['scipy==1.7.3', 'numpy==1.21.6']
import torch


def log_ndtr(input_tensor: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from scipy.stats import norm

    output = torch.from_numpy(norm.logcdf(input_tensor.numpy()))
    # Ensure the output has the same dtype as the input
    return output.to(dtype=input_tensor.dtype)
