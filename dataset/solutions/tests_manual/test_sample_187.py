import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_187 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_187 import custom_function
from sympy import symbols, Eq, sin, cos, exp, log, sqrt, pi, I, S

x, y = symbols("x y")
eq = Eq(x, y)
output = custom_function(eq)

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    expect = eq.lhs - eq.rhs
    assert output == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
