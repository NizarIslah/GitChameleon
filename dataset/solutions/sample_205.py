# library: sympy
# version: 1.13
# extra_dependencies: []
import sympy


def custom_totient(n: int) -> int:
    return sympy.totient(n)
