import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

# We add a guard to avoid import errors in environments
# that don't meet Gradio/Matplotlib requirements (e.g., numpy>=1.23).
# If the environment cannot import gradio for that reason, we skip all tests.
try:
    # Check numpy version first
    import numpy
    from packaging import version
    if version.parse(numpy.__version__) < version.parse("1.23"):
        raise ImportError(
            f"Skipping tests because numpy>=1.23 is required, found {numpy.__version__}"
        )
    
    import gradio as gr

    # Only insert parent directory after we confirm imports won't fail
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import sample_40


assert type(iface.input_components[0])==type(gr.inputs.Image()) and type(iface.output_components[0])==type(gr.outputs.Textbox()) or type(iface.input_components[0])==type(gr.components.Image()) and type(iface.output_components[0])==type(gr.components.Textbox())