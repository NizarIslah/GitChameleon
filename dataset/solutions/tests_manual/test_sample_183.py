import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_183 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_183 import custom_check_carmichael
from sympy.ntheory.factor_ import is_carmichael

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

n = 561
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    from sympy import is_carmichael

    expect = is_carmichael(561)
    output = custom_check_carmichael(n)
    assert output == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
