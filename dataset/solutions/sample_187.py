# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import *
import sympy


def custom_function(eq: sympy.Equality) -> sympy.Expr:
    return eq.lhs - eq.rhs
