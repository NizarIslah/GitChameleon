# library: pandas
# version: 2
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd
import numpy as np


def get_expected_value(df: pd.DataFrame) -> pd.Series:
    return pd.Series([98.0, 99.0], index=["book1", "book2"], dtype=np.float64)
