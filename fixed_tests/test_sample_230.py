import pytest
import pathlib
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_230

def test_pytest_pycollect_makemodule_exists():
    """Test that the pytest_pycollect_makemodule hook exists."""
    assert hasattr(sample_230, 'pytest_pycollect_makemodule')
    assert callable(sample_230.pytest_pycollect_makemodule)

def test_pytest_pycollect_makemodule_signature():
    """Test that the hook has the correct signature."""
    import inspect
    sig = inspect.signature(sample_230.pytest_pycollect_makemodule)
    
    # Check that there's exactly one parameter
    assert len(sig.parameters) == 1
    
    # Check that the parameter name is 'module_path'
    assert 'module_path' in sig.parameters
    
    # Check that the parameter type annotation is pathlib.Path
    param = sig.parameters['module_path']
    assert param.annotation == pathlib.Path

def test_pytest_pycollect_makemodule_is_registered():
    """Test that the hook is properly registered with pytest."""
    # Get the pytest hooks manager
    from _pytest.config import _get_config
    from _pytest.main import _pytest_main

    # Create a pytest config object
    config = _get_config()
    
    # Get all registered hooks
    hooks = config.hook._get_hook_impls()
    
    # Check if our hook is in the registered hooks
    assert 'pytest_pycollect_makemodule' in hooks

def test_pytest_pycollect_makemodule_execution():
    """Test that the hook can be called without errors."""
    # Create a temporary path object
    temp_path = pathlib.Path(__file__)
    
    # Call the hook function - it should not raise any exceptions
    result = sample_230.pytest_pycollect_makemodule(temp_path)
    
    # The current implementation returns None (pass statement)
    assert result is None
