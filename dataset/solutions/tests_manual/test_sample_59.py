import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_59 import get_expected_value

df = pd.DataFrame({"price": [11.1, 12.2]}, index=["book1", "book2"])
original_prices = df["price"]
new_prices = np.array([98, 99])
df.iloc[:, 0] = new_prices
correct_prices = pd.Series([98.0, 99.0], index=["book1", "book2"], dtype=np.float64)
assert get_expected_value(df).equals(correct_prices)
