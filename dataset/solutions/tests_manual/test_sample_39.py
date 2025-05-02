#!/usr/bin/env python
# test_sample.py
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from sample_39 import iface
assert type(gr.Image()) == type(iface.output_components[0])