import math

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_12 import log_ndtr

input_tensor = torch.linspace(-10, 10, steps=20)
expected_result = torch.special.log_ndtr(input_tensor)
assert torch.allclose(log_ndtr(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
