# Add the parent directory to import sys
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_13 import invert_mask

tensor1 = torch.Tensor([1, 2, 3])
tensor2 = torch.Tensor([3, 1, 2])
expected_mask = torch.Tensor([False, True, True])
assert torch.all(torch.eq(invert_mask(tensor1, tensor2), expected_mask))
