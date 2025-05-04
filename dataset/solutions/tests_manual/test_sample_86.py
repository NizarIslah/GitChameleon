import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_86
import unittest
import lightgbm as lgb
import numpy as np
from sample_86 import get_params

data = np.random.rand(10, 2)
label = np.random.randint(2, size=10)
dataset = lgb.Dataset(data, label=label)

params = get_params(dataset)
assert isinstance(params, dict) or params is None
