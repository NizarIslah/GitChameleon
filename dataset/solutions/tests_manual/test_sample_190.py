import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_190 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_190 import custom_body
from sympy.physics.mechanics import RigidBody, Particle

rigid_body_text = "rigid_body"
particle_text = "particle"
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SymPyDeprecationWarning)
    exp1, exp2 = custom_body(rigid_body_text, particle_text)
    assert exp1.name == rigid_body_text
    assert exp2.name == particle_text
    assert not any(
        isinstance(warn.message, SymPyDeprecationWarning) for warn in w
    ), "Test Failed: Deprecation warning was triggered!"
