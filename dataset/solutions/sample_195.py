# library: sympy
# version: 1.10
# extra_dependencies: []
import sympy


def custom_bottom_up(expr: sympy.Expr) -> int:
    return sympy.bottom_up(expr, lambda x: x.doit())
