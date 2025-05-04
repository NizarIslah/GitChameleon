import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_192 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_192 import custom_create_matrix
from sympy import Matrix, symbols


first = [1, 2]
second = [3, 4]
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)

    expected_shape = (2, 2)
    expected_content: list[list[int]] = [[1, 2], [3, 4]]
    output = custom_create_matrix(first, second)

    assert output.shape == expected_shape

    assert output.tolist() == expected_content
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
