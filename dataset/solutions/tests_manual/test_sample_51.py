import unittest
import sys
import os

# Add the parent directory to sys.path to import the module to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_51 import get_scorer_names

conditions = isinstance(get_scorer_names(), list) and len(get_scorer_names()) > 0
assert conditions
