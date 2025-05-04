# Add the parent directory to import sys
import os
import sys
import unittest

import sympy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_179 import custom_preorder_traversal

expr = sympy.Add(1, sympy.Mul(2, 3))
expect = [7]
assert list(custom_preorder_traversal(expr)) == expect
