import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_63 import combined

series1 = pd.Series([1, 2])
series2 = pd.Series([3, 4])
df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list("AB"))
expected_series_values = [1, 2, 3, 4]
expected_dataframe_values = [[1, 2], [3, 4], [5, 6], [7, 8]]
combined_dataframe, combined_series = combined(df1, df2, series1, series2)
assert (
    list(combined_series) == expected_series_values
), "Combined series values are incorrect"
assert (
    combined_dataframe.values.tolist() == expected_dataframe_values
), "Combined dataframe values are incorrect"
