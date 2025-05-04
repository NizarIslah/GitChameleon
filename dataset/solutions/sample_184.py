# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import *


def custom_function(n: int, k: int) -> int:
    return divisor_sigma(n, k)
