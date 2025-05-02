# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_9 import bessel_i0
from scipy.special import i0 as scipy_i0

input_tensor = torch.linspace(0, 10, steps=10)
expected_result = torch.Tensor([1.0000e+00,1.3333e+00,2.6721e+00,6.4180e+00,1.6648e+01,4.4894e+01,1.2392e+02,3.4740e+02,9.8488e+02,2.8157e+03])
assert torch.allclose(bessel_i0(input_tensor), expected_result, rtol=1e-3, atol=1e-3)