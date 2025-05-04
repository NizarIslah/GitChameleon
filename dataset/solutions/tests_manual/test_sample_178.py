import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sympy
from sample_178 import custom_trace

from sympy.physics.quantum.trace import Tr

expect = Tr(2)
assert custom_trace(2) == expect
