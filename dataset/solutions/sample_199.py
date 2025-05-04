# library: sympy
# version: 1.11
# extra_dependencies: []
import sympy


def custom_divides(n: int, p: int) -> bool:
    return n % p == 0
