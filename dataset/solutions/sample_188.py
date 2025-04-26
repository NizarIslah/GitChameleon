# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import symbols, Poly
import sympy

def custom_generatePolyList(poly: sympy.Poly) -> list[int]:
    return p.rep.to_list()