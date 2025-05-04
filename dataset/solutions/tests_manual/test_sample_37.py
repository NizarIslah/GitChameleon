# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_37


assert (
    not render_quadratic_formula().startswith("$")
    and not render_quadratic_formula().endswith("$")
    and "$" in interface.latex_delimiters[0]
    and "$" in interface.latex_delimiters[1]
)
