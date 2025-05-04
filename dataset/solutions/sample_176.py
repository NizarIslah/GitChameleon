# library: sympy
# version: 1.9
# extra_dependencies: []
import sympy
from sympy.matrices.expressions.fourier import DFT


def custom_computeDFT(n: int) -> sympy.ImmutableDenseMatrix:
    return DFT(n).as_explicit()
