import pytest
import pathlib
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_228


def test_pytest_ignore_collect_exists():
    """Test that the pytest_ignore_collect hook function exists."""
    assert hasattr(sample_228, "pytest_ignore_collect")
    assert callable(sample_228.pytest_ignore_collect)


def test_pytest_ignore_collect_accepts_path_parameter():
    """Test that pytest_ignore_collect accepts a pathlib.Path parameter."""
    # Create a test path
    test_path = pathlib.Path("/some/test/path")

    # Call the function with the test path
    # This should not raise any exceptions if the parameter type is correct
    result = sample_228.pytest_ignore_collect(test_path)

    # The function currently returns None (pass), so we just verify it doesn't raise an exception
    assert result is None


def test_pytest_ignore_collect_signature():
    """Test that the function has the correct signature for a pytest hook."""
    import inspect


import inspect


def test_pytest_ignore_collect_signature():
    sig = inspect.signature(pytest_ignore_collect)
    params = list(sig.parameters.items())
    name, param = params[0]
    expect = pathlib.Path
    assert param.annotation == expect


test_pytest_ignore_collect_signature()
