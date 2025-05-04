# library: sympy
# version: 1.13
# extra_dependencies: []
import sympy


def custom_npartitions(n: int) -> int:
    return sympy.functions.combinatorial.numbers.partition(n)
