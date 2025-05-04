import unittest
import operator
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_97 import accumulate_functional

iterable = [1, 2, 3, 4, 5]
func = operator.add
result = accumulate_functional(iterable, func)
assert isinstance(result, list)
assert result == [1, 3, 6, 10, 15]
