import json
import os
import sys
import tempfile
import unittest

from flask import Flask

# Import the function to test

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_148 import app, load_config

load_config(config_file)
assertion_result = app.config["DEBUG"] is True and app.config["SECRET_KEY"] == "secret"
assert assertion_result
