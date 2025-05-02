import os
import sys
import unittest
import warnings

# We attempt to import Gradio and sample_38,
# but if the environment is missing compatible versions (e.g., numpy>=1.23),
# we'll skip these tests to avoid import errors.
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import sample_38


assert iface.output_components[0].show_share_button==False