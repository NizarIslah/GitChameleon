import pytest
import torch
import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.stats import norm
from sample_0 import log_ndtr

from scipy.stats import norm

input_tensor = torch.linspace(-10, 10, steps=20)
expected_result = torch.from_numpy(norm.logcdf(input_tensor.numpy()))
assert torch.allclose(log_ndtr(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
