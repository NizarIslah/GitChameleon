# library: sympy
# version: 1.11
# extra_dependencies: []
import sympy


def custom_is_perfect_square(n: int) -> bool:
    return sympy.ntheory.primetest.is_square(n)
