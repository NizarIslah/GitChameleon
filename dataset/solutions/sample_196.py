# library: sympy
# version: 1.10
# extra_dependencies: []
import sympy


def custom_use(expr: sympy.Expr) -> int:
    return sympy.use(expr, lambda x: x.doit())
