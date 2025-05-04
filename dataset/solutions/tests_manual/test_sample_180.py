# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_180 import custom_parse_mathematica
from sympy import Integer, Max, Min, symbols

expr = "F[6,4,4]"
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    expect = 24
    assert custom_parse_mathematica(expr) == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
