#!/usr/bin/env python
# test_sample.py
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_39


assert type(gr.Image()) == type(iface.output_components[0])
