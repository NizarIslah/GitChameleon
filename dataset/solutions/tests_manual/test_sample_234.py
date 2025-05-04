import pytest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_234 import foo

import dis
import inspect


def test_assert_in_test_foo_bytecode():
    original_test_foo = inspect.unwrap(test_foo)
    instructions = list(dis.get_instructions(original_test_foo))
    has_raise = any(instr.opname == "RAISE_VARARGS" for instr in instructions)
    assert has_raise


test_assert_in_test_foo_bytecode()
