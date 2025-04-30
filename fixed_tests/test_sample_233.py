# Test file for sample_233.py
import pytest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_233 import CustomItem

class TestCustomItem:
    def test_init_with_additional_arg(self):
        """Test that CustomItem initializes with additional_arg parameter."""
        # Create a mock parent using pytest.Module
        parent = pytest.Module.from_parent(parent=None, path=__file__)
        
        # Create a CustomItem instance with the required additional_arg
        test_value = "test_value"
        item = CustomItem.from_parent(parent=parent, name="test_item", additional_arg=test_value)
        
        # Verify that the additional_arg was stored correctly
        assert item.additional_arg == test_value
        
    def test_init_requires_additional_arg(self):
        """Test that CustomItem requires additional_arg parameter."""
        # Create a mock parent using pytest.Module
        parent = pytest.Module.from_parent(parent=None, path=__file__)
        
        # Verify that omitting additional_arg raises TypeError
        with pytest.raises(TypeError):
            CustomItem.from_parent(parent=parent, name="test_item")
            
    def test_inheritance(self):
        """Test that CustomItem inherits from pytest.Item."""
        # Verify inheritance
        assert issubclass(CustomItem, pytest.Item)
