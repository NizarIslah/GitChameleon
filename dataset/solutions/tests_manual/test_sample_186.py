import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_186 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_186 import custom_generateInertia
from sympy import symbols, S, simplify
from sympy.physics.mechanics import ReferenceFrame
import sympy.physics.vector


N = ReferenceFrame("N")
Ixx, Iyy, Izz = symbols("Ixx Iyy Izz")
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    from sympy.physics.mechanics import inertia

    expect = inertia(N, Ixx, Iyy, Izz)
    assert custom_generateInertia(N, Ixx, Iyy, Izz) == expect
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
