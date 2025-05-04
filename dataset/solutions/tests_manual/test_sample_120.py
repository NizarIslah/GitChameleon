import os

# Add the parent directory to the path so we can import the solution module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_120 import compute_moment
from scipy.stats import expon, norm, uniform


from scipy.stats import norm
import numpy as np

dist = norm(15, 10)
n = 5
output = compute_moment(dist, n=n)
assertion_value = np.allclose(output, dist.moment(order=n))
assert assertion_value
