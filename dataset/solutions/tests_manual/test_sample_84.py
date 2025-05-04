import unittest
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_84

assert {
    "train binary_logloss-mean",
    "train binary_logloss-stdv",
    "valid binary_logloss-mean",
    "valid binary_logloss-stdv",
}.issubset(cv_results.keys())
