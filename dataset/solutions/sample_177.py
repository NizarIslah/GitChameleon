# library: sympy
# version: 1.9
# extra_dependencies: []
from typing import Tuple
from sympy import laplace_transform, symbols, eye
import sympy


def custom_laplace_transform(
    t: sympy.Symbol, z: sympy.Symbol
) -> Tuple[sympy.Matrix, sympy.Expr, bool]:
    return laplace_transform(eye(2), t, z, legacy_matrix=False)
