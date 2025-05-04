import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_188 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_188 import custom_generatePolyList
from sympy import symbols, Poly


import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

x = symbols("x")
p = Poly(x**2 + 2 * x + 3)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    expect = [1, 2, 3]
    assert custom_generatePolyList(p) == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
