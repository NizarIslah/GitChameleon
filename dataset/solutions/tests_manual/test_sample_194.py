import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_194 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_194 import custom_function
from sympy import Matrix, symbols

m = Matrix([[1, 2], [3, 4]])

output = custom_function(m)
output[(0, 0)] = 100

assert m[0, 0] == 1
assert output[(0, 0)] == 100
