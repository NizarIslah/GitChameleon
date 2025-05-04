# library: sympy
# version: 1.9
# extra_dependencies: []
from sympy import Matrix
import sympy


def custom_create_matrix(first: sympy.Matrix, second: sympy.Matrix) -> list[int]:
    return Matrix([first, second])
