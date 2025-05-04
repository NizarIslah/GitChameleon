import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_185 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_185 import custom_function
from sympy import GF
from sympy.polys.domains.finitefield import FiniteField

K = GF(6)
a = K(8)
output = custom_function(K, a)

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    expect = 2
    assert output == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
