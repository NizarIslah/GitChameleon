# Test file for sample_233.py
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_233 import CustomItem

import inspect

signature = inspect.signature(CustomItem.__init__)
assert any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values())
