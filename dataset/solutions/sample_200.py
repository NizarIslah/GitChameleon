# library: sympy
# version: 1.12
# extra_dependencies: []
from sympy import Matrix, symbols, Array
import sympy


def custom_array_to_matrix(array: sympy.Array) -> sympy.Matrix:
    from sympy.tensor.array.expressions.from_array_to_matrix import (
        convert_array_to_matrix,
    )

    return convert_array_to_matrix(array)
