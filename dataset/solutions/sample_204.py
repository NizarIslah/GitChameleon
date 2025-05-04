# library: sympy
# version: 1.13
# extra_dependencies: []
import sympy


def custom_prime_counting(n: int) -> int:
    return sympy.primepi(n)
