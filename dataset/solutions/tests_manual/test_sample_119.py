import os
import sys
import unittest

import numpy as np
from scipy.stats import expon, norm, uniform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_119 import compute_moment


from scipy.stats import norm
import numpy as np

dist = norm(15, 10)
n = 5
output = compute_moment(dist, n=n)
assertion_value = np.allclose(output, dist.moment(order=n))
assert assertion_value
