import os
import sys
import unittest

from jinja2 import Environment
from markupsafe import Markup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_155 import get_output, nl2br_core, solution

nl2br = solution()
env = Environment(autoescape=True)
output = get_output(env, nl2br)
expected = "<br>Hello</br> World"

assert output == expected, f"Expected: {expected!r}, but got: {output!r}"
