# library: sympy
# version: 1.13
# extra_dependencies: []
import sympy


def custom_jacobi_symbols(a: int, n: int) -> int:
    return sympy.jacobi_symbol(a, n)
