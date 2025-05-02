import os
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_324 import sol_dict
assert sol_dict['total'] == float('inf')