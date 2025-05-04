import os
import sys
import unittest

import jinja2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_154 import setup_environment, solution

greet = solution()
env = setup_environment("greet", greet)
template = env.from_string(
    """
{{ 'World'| greet }}"""
)
assertion_results = "Hi, World!" in template.render(prefix="Hi")
assert assertion_results
assertion_results = "Hello, World!" in template.render()
assert assertion_results
