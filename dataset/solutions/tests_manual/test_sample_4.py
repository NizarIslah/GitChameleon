# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_4 import bessel_i0
from scipy.special import i0 as scipy_i0

input_tensor = torch.linspace(0, 10, steps=10)
expected_result = torch.Tensor(
    [
        1.0000e00,
        1.3333e00,
        2.6721e00,
        6.4180e00,
        1.6648e01,
        4.4894e01,
        1.2392e02,
        3.4740e02,
        9.8488e02,
        2.8157e03,
    ]
)
assert torch.allclose(bessel_i0(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
