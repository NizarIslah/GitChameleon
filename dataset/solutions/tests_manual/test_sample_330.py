# Test for sample_330.py
import os
import sys
import unittest

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_330 import use_seaborn

use_seaborn()

cycle = plt.rcParams["axes.prop_cycle"]
from cycler import cycler

a = cycler("color", ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"])
assert cycle == a
