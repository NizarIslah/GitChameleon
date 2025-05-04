# library: sympy
# version: 1.11
# extra_dependencies: []
import sympy


def custom_is_prime(n: int) -> bool:
    return sympy.isprime(n)
