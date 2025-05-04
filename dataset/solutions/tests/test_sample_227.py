import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_227


def test_pytest_runtest_setup_hook_exists():
    """Test that the pytest_runtest_setup hook function exists."""
    assert hasattr(sample_227, "pytest_runtest_setup")
    assert callable(sample_227.pytest_runtest_setup)


def test_pytest_runtest_setup_yields():
    """Test that the pytest_runtest_setup function yields control."""
    # Create a generator from the hook function
    generator = sample_227.pytest_runtest_setup()

    # Verify it's a generator
    assert hasattr(generator, "__iter__")
    assert hasattr(generator, "__next__")

    # Verify it yields (and doesn't raise an exception)
    try:
        next(generator)
        # After yield, the function should be done
        with pytest.raises(StopIteration):
            next(generator)
    except Exception as e:
        pytest.fail(f"The hook function raised an exception: {e}")


def test_pytest_runtest_setup_integration():
    """Test the hook in a more realistic scenario by mocking the pytest hook system."""
    # Create a mock for the hook caller
    mock_hook_caller = MagicMock()

    # Mock the pytest hooks
    with patch("pytest.hookimpl") as mock_hookimpl:
        # Create a generator from our hook
        gen = sample_227.pytest_runtest_setup()

        # Simulate what pytest would do: get to the yield point
        next(gen)

        # At this point, pytest would run other setup hooks
        # Then it would continue our generator
        try:
            # This should raise StopIteration since our generator is done after yield
            next(gen)
            pytest.fail("Generator should be exhausted after yield")
        except StopIteration:
            # This is expected
            pass
