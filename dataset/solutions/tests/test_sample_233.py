# Test file for sample_233.py
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_233 import CustomItem


class TestCustomItem:
    def test_init_requires_additional_arg(self):
        """Test that CustomItem requires additional_arg parameter."""
        # We need to create a mock parent and name since pytest.Item requires these
        parent_mock = MagicMock()
        parent = pytest.Module.from_parent(parent=parent_mock, path=Path(__file__))

        # Verify that omitting additional_arg raises TypeError
        with pytest.raises(TypeError):
            CustomItem.from_parent(parent=parent, name="test_item")

    def test_inheritance(self):
        """Test that CustomItem inherits from pytest.Item."""
        # Verify inheritance
        assert issubclass(CustomItem, pytest.Item)
