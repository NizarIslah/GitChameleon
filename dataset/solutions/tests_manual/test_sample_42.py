# File: test_gradio_app.py

import unittest
import gradio as gr
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_42 as gradio_app_module


assert (
    type(iface.input_components[0]) == gr.Dropdown
    and iface.input_components[0].multiselect == True
) or type(iface.input_components[0]) == gr.CheckboxGroup
