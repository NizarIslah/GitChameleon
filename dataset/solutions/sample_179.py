# library: sympy
# version: 1.11
# extra_dependencies: []
import sympy


def custom_preorder_traversal(expr: sympy.Expr) -> sympy.core.basic.preorder_traversal:
    return sympy.preorder_traversal(expr)
