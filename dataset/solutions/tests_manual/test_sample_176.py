import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
import sympy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_176 import custom_computeDFT

import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning


def test_custom_computeDFT():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", SymPyDeprecationWarning)  # Capture all warnings
        output = custom_computeDFT(4)
        expect = DFT(4).as_explicit()
        assert output == expect
        assert not any(
            isinstance(warn.message, SymPyDeprecationWarning) for warn in w
        ), "Test Failed: Deprecation warning was triggered!"


test_custom_computeDFT()
