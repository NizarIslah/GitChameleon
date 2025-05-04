# library: sympy
# version: 1.13
# extra_dependencies: []
import sympy


def custom_legendre(a: int, n: int) -> int:
    return sympy.legendre_symbol(a, n)
