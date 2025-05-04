import pytest
import pathlib
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_230


def test_pytest_pycollect_makemodule_exists():
    """Test that the pytest_pycollect_makemodule hook exists."""
    assert hasattr(sample_230, "pytest_pycollect_makemodule")
    assert callable(sample_230.pytest_pycollect_makemodule)


def test_pytest_pycollect_makemodule_signature():
    """Test that the hook has the correct signature."""
    import inspect


import inspect


def test_pytest_pycollect_makemodule_signature():
    sig = inspect.signature(pytest_pycollect_makemodule)
    params = list(sig.parameters.items())
    name, param = params[0]
    expect = pathlib.Path
    assert param.annotation == expect


test_pytest_pycollect_makemodule_signature()
