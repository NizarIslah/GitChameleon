# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_2 import erf
from scipy.special import erf as scipy_erf

input_tensor = torch.linspace(0, 10, steps=10)
expected_result = torch.Tensor(
    [0.0000, 0.8839, 0.9983, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
)
assert torch.allclose(erf(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
