import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_191 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_191 import custom_symbol
from sympy import Indexed, Symbol, IndexedBase

a = Indexed("A", 0)
output = custom_symbol(a)

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    expect = a.free_symbols
    assert output == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
