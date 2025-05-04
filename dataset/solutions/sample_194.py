# library: sympy
# version: 1.9
# extra_dependencies: []
from sympy import Matrix
import sympy


def custom_function(matrix: sympy.Matrix) -> list[int]:
    return matrix.todok()
