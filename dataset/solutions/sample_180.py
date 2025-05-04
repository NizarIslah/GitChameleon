# library: sympy
# version: 1.11
# extra_dependencies: []
from sympy.parsing.mathematica import parse_mathematica
from sympy import Function, Max, Min
import sympy


def custom_parse_mathematica(expr: str) -> int:
    return parse_mathematica(expr).replace(Function("F"), lambda *x: Max(*x) * Min(*x))
