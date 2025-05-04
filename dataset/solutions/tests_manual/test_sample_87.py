# library: lightgbm
# version: 3.0.0
# extra_dependencies: ['numpy==1.26.4']
import unittest
import numpy as np
import json
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_87 import dump_json

NUMPY_ARRAY = np.array([1, 2, 3])
json_data = dump_json(NUMPY_ARRAY)
expected = "[1, 2, 3]"
assert json_data == expected
