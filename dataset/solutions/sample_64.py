# library: pandas
# version: 2
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd


def correct_type(index: pd.Index) -> str:
    return str(index.dtype)
