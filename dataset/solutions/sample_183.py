# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import *


def custom_check_carmichael(n: int) -> bool:
    return is_carmichael(n)
