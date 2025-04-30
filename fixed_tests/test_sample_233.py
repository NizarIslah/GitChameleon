import pathlib
import pytest
import sys
import os

# Ensure the parent directory is in sys.path to find sample_233.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_233 import CustomItem

class TestCustomItem:
    def test_init_with_additional_arg(self):
        """Test that CustomItem initializes with additional_arg parameter."""
        # Create a temporary Pytest session and a Module as parent
        config = pytest.Config()
        session = pytest.Session.from_config(config)
        parent = pytest.Module.from_parent(parent=session, path=pathlib.Path(__file__))

        # Create a CustomItem instance with the required additional_arg
        test_value = "test_value"
        item = CustomItem.from_parent(parent=parent, name="test_item", additional_arg=test_value)

        # Verify that the additional_arg was stored correctly
        assert item.additional_arg == test_value

    def test_init_requires_additional_arg(self):
        """Test that CustomItem requires additional_arg parameter."""
        # Create a temporary Pytest session and a Module as parent
        config = pytest.Config()
        session = pytest.Session.from_config(config)
        parent = pytest.Module.from_parent(parent=session, path=pathlib.Path(__file__))

        # Verify that omitting additional_arg raises TypeError
        with pytest.raises(TypeError):
            CustomItem.from_parent(parent=parent, name="test_item")

    def test_inheritance(self):
        """Test that CustomItem inherits from pytest.Item."""
        # Verify inheritance
        assert issubclass(CustomItem, pytest.Item)