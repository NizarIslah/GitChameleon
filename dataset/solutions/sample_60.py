# library: pandas
# version: 1.5.0
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd
import numpy as np


def get_slice(ser: pd.Series, start: int, end: int) -> pd.Series:
    return ser[start:end]
