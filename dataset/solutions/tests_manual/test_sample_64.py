import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_64 import correct_type

index = pd.Index([1, 2, 3], dtype="int32")
assert isinstance(correct_type(index), str)
assert correct_type(index) == str(index.dtype)
