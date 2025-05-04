import os

# Add the parent directory to the path so we can import the solution
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
from sample_325 import compute_scattering

import kymatio

a = torch.ones((1, 3, 32, 32))
S, S_a = compute_scattering(a)
assert isinstance(S_a, torch.Tensor)
assert isinstance(S, kymatio.scattering2d.frontend.torch_frontend.ScatteringTorch2D)
