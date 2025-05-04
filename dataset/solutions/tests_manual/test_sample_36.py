# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_36


assert render_quadratic_formula().startswith(
    "$"
) and render_quadratic_formula().endswith("$")
