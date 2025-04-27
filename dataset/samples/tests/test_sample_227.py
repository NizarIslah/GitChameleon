import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.solutions.sample_227 import pytest_runtest_setup


def test_pytest_runtest_setup_hook_registered():
    """Test that the pytest_runtest_setup hook is properly registered."""
    # Get all registered hooks for pytest_runtest_setup
    hooks = pytest.hook.pytest_runtest_setup._nonwrappers + pytest.hook.pytest_runtest_setup._wrappers
    
    # Check if our hook is in the registered wrappers
    hook_found = any(h.function is pytest_runtest_setup for h in pytest.hook.pytest_runtest_setup._wrappers)
    
    assert hook_found, "pytest_runtest_setup hook was not properly registered"


def test_pytest_runtest_setup_yields():
    """Test that the pytest_runtest_setup hook yields once."""
    # Create a generator from the hook function
    generator = pytest_runtest_setup()
    
    # The hook should yield once
    next(generator)
    
    # After yielding once, it should be exhausted
    with pytest.raises(StopIteration):
        next(generator)


def test_pytest_runtest_setup_is_hookwrapper():
    """Test that pytest_runtest_setup is marked as a hookwrapper."""
    # Check if the function has the hookwrapper marker
    assert hasattr(pytest_runtest_setup, 'hookwrapper')
    assert pytest_runtest_setup.hookwrapper is True


@patch('dataset.solutions.sample_227.pytest_runtest_setup')
def test_hook_called_during_test_execution(mock_hook):
    """Test that the hook is called during test execution."""
    # Setup the mock to return a generator that yields once
    mock_generator = MagicMock()
    mock_generator.__iter__.return_value = iter([None])
    mock_hook.return_value = mock_generator
    
    # Run a simple test function
    @pytest.mark.xfail(reason="This test is expected to fail as it's just for hook testing")
    def dummy_test():
        assert False
    
    # Execute the test
    pytest.main(['-xvs', '--no-header', '--no-summary'])
    
    # Verify the hook was called
    mock_hook.assert_called()