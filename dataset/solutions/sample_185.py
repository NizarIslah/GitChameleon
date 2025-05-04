# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import GF
from sympy.polys.domains.finitefield import FiniteField


def custom_function(K: FiniteField, a: FiniteField) -> int:
    return K.to_int(a)
