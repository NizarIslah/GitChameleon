# library: kymatio
# version: 0.3.0
# extra_dependencies: ['torch==1.4.0']
import kymatio
import torch
from kymatio import Scattering2D
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
from typing import Tuple


def compute_scattering(a: torch.Tensor) -> Tuple[torch.Tensor, ScatteringTorch2D]:
    S = Scattering2D(2, (32, 32), frontend="torch")
    S_a = S(a)
    return S, S_a
