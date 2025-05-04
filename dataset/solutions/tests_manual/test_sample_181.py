import os
import sys
import unittest

from sympy.physics.mechanics import Body, PinJoint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_181 import custom_pinJoint

parent, child = Body("parent"), Body("child")
pin = custom_pinJoint(parent, child)
expect1 = parent.frame.x
expect2 = -child.frame.x

assert pin.parent_point.pos_from(parent.masscenter) == expect1
assert pin.child_point.pos_from(child.masscenter) == expect2
