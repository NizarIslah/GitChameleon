import unittest

import sympy
from sample_177 import custom_laplace_transform
from sympy import Matrix, eye, symbols

t, z = symbols("t z")
from sympy import Matrix

output = custom_laplace_transform(t, z)
expected = (Matrix([[1 / z, 0], [0, 1 / z]]), 0, True)
assert output == expected
