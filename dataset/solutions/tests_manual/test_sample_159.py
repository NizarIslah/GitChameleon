import os
import sys
import unittest

import numpy as np
from scipy.stats import hmean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_159 import count_unique_hmean


data = np.array(
    [[1, 2, 3], [2, 2, 2], [1, np.nan, 3], [4, 5, 6], [np.nan, 1, np.nan], [1, 2, 3]]
)
assertion_value = count_unique_hmean(data) == 5
assert assertion_value
