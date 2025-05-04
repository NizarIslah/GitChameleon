import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_202 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_202 import custom_npartitions
import sympy

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)

    expect = 7
    output = custom_npartitions(5)
    assert output == expect

    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
