import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_113 import combine_pvalues


A = np.array(
    [
        0.01995382,
        0.1906752,
        0.71157923,
        0.44477942,
        0.4535412,
        0.67556953,
        0.11174941,
        0.85494112,
        0.33214635,
        0.19103228,
    ]
)
output = combine_pvalues(A)
assertion_value = np.allclose(
    np.asarray(output),
    np.asarray(
        [
            -stats.combine_pvalues(1 - A, "fisher")[0],
            (1 - stats.combine_pvalues(1 - A, "fisher")[1]),
        ]
    ),
)
assert assertion_value
