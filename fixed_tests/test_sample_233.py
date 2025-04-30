import pytest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_233 import CustomItem

class TestCustomItem:
    @pytest.fixture
    def parent_module(self):
        """
        Create a proper pytest "session" and "Module" parent so that
        pytest.Item.from_parent can be called without errors.
        """
        plugin_manager = pytest.PytestPluginManager()
        config = pytest.Config.fromdictargs({}, plugin_manager)
        session = pytest.Session.from_config(config)
        mod = pytest.Module.from_parent(session, path=__file__)
        return mod

    def test_init_with_additional_arg(self, parent_module):
        """Test that CustomItem initializes with the additional_arg parameter."""
        test_value = "test_value"
        item = CustomItem.from_parent(
            parent=parent_module,
            name="test_item",
            additional_arg=test_value
        )
        assert item.additional_arg == test_value

    def test_init_requires_additional_arg(self, parent_module):
        """Test that CustomItem requires the additional_arg parameter."""
        with pytest.raises(TypeError):
            CustomItem.from_parent(
                parent=parent_module,
                name="test_item"
            )

    def test_inheritance(self):
        """Test that CustomItem inherits from pytest.Item."""
        assert issubclass(CustomItem, pytest.Item)