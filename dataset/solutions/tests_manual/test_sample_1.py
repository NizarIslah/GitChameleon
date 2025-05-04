# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_1 import gamma_ln
from scipy.special import gammaln as scipy_gammaln

input_tensor = torch.linspace(0, 10, steps=10)
expected_result = torch.Tensor(
    [
        float("inf"),
        -0.0545,
        0.1092,
        1.0218,
        2.3770,
        4.0476,
        5.9637,
        8.0806,
        10.3675,
        12.8018,
    ]
)
assert torch.allclose(gamma_ln(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
