import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_184 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_184 import custom_function
from sympy import divisors

n = 6
k = 1
output = custom_function(n, k)
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    expect = 12
    assert output == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
