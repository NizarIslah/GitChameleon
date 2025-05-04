import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_197 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_197 import custom_is_perfect_square

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)

    expect = True
    output = custom_is_perfect_square(4)

    assert output == expect

    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
