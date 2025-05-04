import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_200 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_200 import custom_array_to_matrix
from sympy import Array, Matrix, symbols
from sympy.tensor.array import ImmutableDenseNDimArray

a1, a2, a3, a4 = symbols("a1 a2 a3 a4")
array_expr = Array([[a1, a2], [a3, a4]])

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    from sympy.tensor.array.expressions.from_array_to_matrix import (
        convert_array_to_matrix,
    )

    expect = convert_array_to_matrix(array_expr)
    output = custom_array_to_matrix(array_expr)

    assert output == expect

    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
