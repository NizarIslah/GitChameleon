# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import contextlib
import io

import nltk
from sample_26 import show_usage
assert "LazyModule supports the following operations" in show_usage(nltk.corpus)
